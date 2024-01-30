""" TODO: Write documentation.
"""
# pylint: disable=attribute-defined-outside-init,invalid-name
import io
import os
import re
import sys
import math
import array
import signal
import warnings
import importlib
import threading
import multiprocessing as mp
import xml.etree.ElementTree as ET
from weakref import ref
from functools import wraps
from itertools import chain
from datetime import datetime
from types import TracebackType
from traceback import TracebackException
from pathlib import PureWindowsPath
from contextlib import AbstractContextManager
from typing import (
    Dict, Any, Callable, Optional, Tuple, Union, Sequence, Iterable, Literal,
    Type)

import numpy as np

import simplepbr
from direct.task import Task
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (  # pylint: disable=no-name-in-module
    NodePath, Point3, Vec3, Vec4, Mat4, Quat, LQuaternion, Geom, GeomEnums,
    GeomNode, GeomTriangles, GeomVertexData, GeomVertexArrayFormat,
    GeomVertexFormat, GeomVertexWriter, PNMImage, PNMImageHeader, TextNode,
    OmniBoundingVolume, CompassEffect, BillboardEffect, InternalName, Filename,
    Material, Texture, TextureStage, TransparencyAttrib, PGTop, Camera, Lens,
    PerspectiveLens, OrthographicLens, ShaderAttrib, AntialiasAttrib,
    CollisionNode, CollisionRay, CollisionTraverser, CollisionHandlerQueue,
    ClockObject, GraphicsPipe, GraphicsOutput, GraphicsWindow, DisplayRegion,
    RenderModeAttrib, WindowProperties, FrameBufferProperties, loadPrcFileData)

import panda3d_viewer.viewer_app
from panda3d_viewer import geometry
from panda3d_viewer.viewer_config import ViewerConfig
from panda3d_viewer.viewer_errors import ViewerClosedError, ViewerError

import hppfcl
import pinocchio as pin
from pinocchio.utils import npToTuple
from pinocchio.visualize import BaseVisualizer

from ..geometry import extract_vertices_and_faces_from_geometry


WINDOW_SIZE_DEFAULT = (600, 600)
CAMERA_POS_DEFAULT = [(4.0, -4.0, 1.5), (0, 0, 0.5)]

SKY_TOP_COLOR = (0.53, 0.8, 0.98, 1.0)
SKY_BOTTOM_COLOR = (0.1, 0.1, 0.43, 1.0)

LEGEND_DPI = 400
LEGEND_SCALE_MAX = 0.42
WATERMARK_SCALE_MAX = 0.2
CLOCK_SCALE = 0.1
WIDGET_MARGIN_REL = 0.02

PANDA3D_FRAMERATE_MAX = 40
PANDA3D_REQUEST_TIMEOUT = 30.0


Tuple3FType = Union[Tuple[float, float, float], np.ndarray]
Tuple4FType = Union[Tuple[float, float, float, float], np.ndarray]
ShapeType = Literal[
    'cone', 'box', 'sphere', 'capsule', 'cylinder', 'frame', 'arrow']
FrameType = Union[Tuple[Tuple3FType, Tuple4FType], np.ndarray]


def _signal_guarded(
        signalnum: int,
        handler: Optional[Union[signal.Handlers, Callable[..., Any], int]]
        ) -> Optional[Union[signal.Handlers, Callable[..., Any], int]]:
    """Guard `signal.signal` to make it a no-op outside of main thread instead
    of raising an exception. This typically happens during async rendering.
    """
    if threading.current_thread() is threading.main_thread():
        return signal.signal(signalnum, handler)
    return signal.getsignal(signalnum)


_signal_guarded_module = type(signal)(signal.__name__, signal.__doc__)
_signal_guarded_module.__dict__.update(signal.__dict__)
_signal_guarded_module.__dict__['signal'] = _signal_guarded
Task.signal = _signal_guarded_module


def _sanitize_path(path: str) -> str:
    """Sanitize path on windows to make it compatible with python bindings.

    `Assimp` bindings used to load meshes and other C++ tools handling path
    does not support several features on Windows. First, it does not support
    symlinks, then the hard drive prefix must be `/x/` instead of `X:\\`,
    folder's name must respect the case, and backslashes must be used as
    delimiter instead of forward slashes.

    :param path: Path to sanitize.
    """
    if sys.platform.startswith('win'):
        path = os.path.realpath(path)
        path = PureWindowsPath(path).as_posix()
        path = re.sub(r'^([A-Za-z]):',
                      lambda m: "/" + m.group(1).lower(),
                      path)
    return path


def make_gradient_skybox(sky_color: Tuple4FType,
                         ground_color: Tuple4FType,
                         span: float = 1.0,
                         offset: float = 0.0,
                         subdiv: int = 2) -> NodePath:
    """Simple gradient to be used as skybox.

    .. seealso::
        https://discourse.panda3d.org/t/color-gradient-scene-background/26946/14

    :param sky_color: Color at zenith as a normalized 4-tuple (R, G, B, A).
    :param ground_color: Color at nadir as a normalized 4-tuple (R, G, B, A).
    :param span: Span of the gradient from ground color to sky color. The color
                 is flat outside range angle [-span/2-offset, span/2-offset].
                 Optional: 1.0 by default.
    :param offset: Offset angle defining the position of the horizon.
                   Optional: 0.0 by default.
    :param subdiv: Number of sub-division for the complete gradient.
                   Optional: 2 by default.
    """  # noqa: E501
    # Check validity of arguments
    assert subdiv > 1, "Number of sub-division must be strictly larger than 1."
    assert 0.0 <= span <= 1.0, "Offset must be in [0.0, 1.0]."

    # Define vertex format
    vformat = GeomVertexFormat()
    aformat = GeomVertexArrayFormat()
    aformat.add_column(
        InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    vformat.add_array(aformat)
    aformat = GeomVertexArrayFormat()
    aformat.add_column(
        InternalName.make("color"), 4, Geom.NT_uint8, Geom.C_color)
    vformat.add_array(aformat)
    vformat = GeomVertexFormat.register_format(vformat)

    # Create a simple, horizontal prism.
    # Make it very wide to avoid ever seeing its left and right sides.
    # One edge is at the "horizon", while the two other edges are above
    # and a bit behind the camera so they are only visible when looking
    # straight up. ((-50., 86.6) -> (cos(2*pi/3), sin(2*pi/3)))
    vertex_data = GeomVertexData(
        "prism_data", vformat, GeomEnums.UH_static)
    vertex_data.unclean_set_num_rows(4 + subdiv * 2)
    values = array.array("f", (-1000., -50., 86.6, 1000., -50., 86.6))
    offset_angle = np.pi / 1.5 * (1.0 - span)
    delta_angle = 2. * (np.pi / 1.5 - offset_angle) / (subdiv + 1)
    for i in range(subdiv):
        angle = np.pi / 3. + offset + offset_angle + delta_angle * (i + 1)
        y = -np.cos(angle) * 100.
        z = np.sin(angle) * 100.
        values.extend((-1000., y, z, 1000., y, z))
    values.extend((-1000., -50., -86.6, 1000., -50., -86.6))
    pos_array = vertex_data.modify_array(0)
    memview = memoryview(pos_array).cast("B").cast("f")
    memview[:] = values

    # Interpolate the colors
    color1 = tuple(int(c * 255) for c in sky_color)
    color2 = tuple(int(c * 255) for c in ground_color)
    values = array.array("B", color1 * 2)
    for ratio in np.linspace(0, 1, subdiv):
        color = tuple(int(c1 * (1 - ratio) + c2 * ratio)
                      for c1, c2 in zip(color1, color2))
        values.extend(color * 2)
    values.extend(color2 * 2)
    color_array = vertex_data.modify_array(1)
    memview = memoryview(color_array).cast("B")
    memview[:] = values

    tris_prim = GeomTriangles(GeomEnums.UH_static)
    indices = array.array("H", (0, 3, 1, 0, 2, 3))
    for i in range(subdiv + 1):
        j = i * 2
        indices.extend((j, 3 + j, 1 + j, j, 2 + j, 3 + j))
    j = (subdiv + 1) * 2
    indices.extend((j, 1, 1 + j, j, 0, 1))
    tris_array = tris_prim.modify_vertices()
    tris_array.unclean_set_num_rows((subdiv + 3) * 6)
    memview = memoryview(tris_array).cast("B").cast("H")
    memview[:] = indices

    # The compass effect can make the node leave its bounds, so make them
    # infinitely large.
    geom = Geom(vertex_data)
    geom.add_primitive(tris_prim)
    node = GeomNode("prism")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    prism = NodePath(node)
    prism.set_color_scale((*(3 * (1.1,)), 1.0))
    prism.set_bin("background", 0)
    prism.set_depth_write(False)
    prism.set_depth_test(False)

    return prism


def make_cone(num_sides: int = 16) -> Geom:
    """Make a close shaped cone, approximate by a pyramid with regular convex
    n-sided polygon base.

    .. seealso::
        For details about regular polygons:
        https://en.wikipedia.org/wiki/Regular_polygon

    :param num_sides: Number of sides for the polygon base.
    """
    # Define vertex format
    vformat = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData('vdata', vformat, Geom.UH_static)
    vdata.unclean_set_num_rows(num_sides + 3)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')

    # Add radial points
    for u in np.linspace(0.0, 2 * np.pi, num_sides + 1):
        x, y = math.cos(u), math.sin(u)
        vertex.addData3(x, y, 0.0)
        normal.addData3(x, y, 0.0)

    # Add top and bottom points
    vertex.addData3(0.0, 0.0, 1.0)
    normal.addData3(0.0, 0.0, 1.0)
    vertex.addData3(0.0, 0.0, 0.0)
    normal.addData3(0.0, 0.0, -1.0)

    # Make triangles.
    # Note that by default, rendering is one-sided. It only renders the outside
    # face, that is defined based on the "winding" order of the vertices making
    # the triangles. For reference, see:
    # https://discourse.panda3d.org/t/procedurally-generated-geometry-and-the-default-normals/24986/2  # noqa: E501  # pylint: disable=line-too-long
    prim = GeomTriangles(Geom.UH_static)
    prim.reserve_num_vertices(6 * num_sides)
    for i in range(num_sides):
        prim.add_vertices(i, i + 1, num_sides + 1)
        prim.add_vertices(i + 1, i, num_sides + 2)

    # Create geometry object
    geom = Geom(vdata)
    geom.add_primitive(prim)

    return geom


def make_pie(theta_start: float = 0.0,
             theta_end: float = 2.0 * math.pi,
             num_segments: int = 16) -> Geom:
    """Make a portion of cylinder along vertical axis, ie a 3D pie chart.

    :param theta_start: Angle at which the filled portion starts.
                        Optional: 0 degree by default.
    :param theta_end: Angle at which the filled portion ends.
                      Optional: 360 degrees by default.
    :param num_segments: Number of segments on the caps.
                         Optional: 16 by default.
    """
    cyl_rows = num_segments * 2
    cap_rows = num_segments + 1
    r0, r1 = cyl_rows, cyl_rows + cap_rows
    is_pie = theta_end - theta_start < 2.0 * math.pi

    vformat = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData('vdata', vformat, Geom.UHStatic)
    vdata.uncleanSetNumRows(cyl_rows + 2 * cap_rows)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')

    # Add radial points
    for phi in np.linspace(theta_start, theta_end, num_segments):
        x, y = math.cos(phi), math.sin(phi)
        for z in (-1, 1):
            vertex.addData3(x, y, z * 0.5)
            normal.addData3(x, y, 0)

    # Add top and bottom points
    for z in (-1, 1):
        vertex.addData3(0, 0, z * 0.5)
        normal.addData3(0, 0, z)
        for phi in np.linspace(theta_start, theta_end, num_segments):
            x, y = math.cos(phi), math.sin(phi)
            vertex.addData3(x, y, z * 0.5)
            normal.addData3(0, 0, z)

    # Make triangles
    prim = GeomTriangles(Geom.UHStatic)
    prim.reserve_num_vertices(4 * (num_segments + int(is_pie)) - 2)
    for i in range(num_segments - 1):
        prim.addVertices(i * 2, i * 2 + 3, i * 2 + 1)
        prim.addVertices(i * 2, i * 2 + 2, i * 2 + 3)

    if is_pie:
        prim.addVertices(r0, r0 + 1, r1 + 1)
        prim.addVertices(r0, r1 + 1, r1)
        prim.addVertices(r0, r1, r1 + num_segments - 1)
        prim.addVertices(r0, r1 + num_segments - 1, r0 + num_segments - 1)

    for i in range(num_segments):
        prim.addVertices(r0, r0 + i + 1, r0 + i)
        prim.addVertices(r1, r1 + i, r1 + i + 1)

    # Create geometry object
    geom = Geom(vdata)
    geom.addPrimitive(prim)

    return geom


def make_torus(minor_radius: float = 0.2, num_segments: int = 16) -> Geom:
    """Make a unit torus geometry which looks like a donut. The distance from
    the axis of revolution called major radius is always 1.0.

    :param minor_radius: The radius of the tube.
    :param num_segments: Number of segments along both the axis of revolution
                         and a slice of the tube.
    """
    vformat = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData('vdata', vformat, Geom.UHStatic)
    vdata.uncleanSetNumRows(num_segments * num_segments)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')

    # Add radial points
    for u in np.linspace(0.0, 2.0 * math.pi, num_segments):
        for v in np.linspace(0.0, 2.0 * math.pi, num_segments):
            x_c, y_c = math.cos(u), math.sin(u)
            x_t = minor_radius * math.cos(v) * math.cos(u)
            y_t = minor_radius * math.cos(v) * math.sin(u)
            z_t = minor_radius * math.sin(v)
            vertex.addData3(x_c + x_t, y_c + y_t, z_t)
            normal.addData3(x_t, y_t, z_t)

    # Make triangles
    prim = GeomTriangles(Geom.UHStatic)
    prim.reserve_num_vertices(2 * (num_segments - 1) ** 2)
    for i in range(num_segments - 1):
        for j in range(num_segments - 1):
            k = i * num_segments + j
            prim.addVertices(k, k + 1, k + num_segments)
            prim.addVertices(k + 1, k + 1 + num_segments, k + num_segments)

    # Create geometry object
    geom = Geom(vdata)
    geom.addPrimitive(prim)

    return geom


class Panda3dApp(panda3d_viewer.viewer_app.ViewerApp):
    """A Panda3D based application.
    """
    def __init__(self,  # pylint: disable=super-init-not-called
                 config: Optional[ViewerConfig] = None) -> None:
        # Enforce viewer configuration
        if config is None:
            config = ViewerConfig()
        config.set_window_size(*WINDOW_SIZE_DEFAULT)
        config.set_window_fixed(False)
        config.enable_antialiasing(True, multisamples=4)
        config.set_value('framebuffer-software', False)
        config.set_value('framebuffer-hardware', False)
        config.set_value('load-display', 'pandagl')
        config.set_value('aux-display',
                         'p3headlessgl'
                         '\naux-display pandadx9'
                         '\naux-display p3tinydisplay')
        config.set_value('window-type', 'offscreen')
        config.set_value('sync-video', False)
        config.set_value('default-near', 0.1)
        config.set_value('gl-version', '3 1')
        config.set_value('notify-level', 'fatal')
        config.set_value('notify-level-x11display', 'fatal')
        config.set_value('notify-level-device', 'fatal')
        config.set_value('default-directnotify-level', 'error')
        loadPrcFileData('', str(config))

        # Define offscreen buffer
        self.buff: Optional[GraphicsOutput] = None

        # Initialize base implementation.
        # Note that the original constructor is by-passed on purpose.
        ShowBase.__init__(self)  # pylint: disable=non-parent-init-called

        # Monkey-patch task manager to ignore SIGINT from keyboard interrupt
        def keyboardInterruptHandler(
                *args: Any, **kwargs: Any  # pylint: disable=unused-argument
                ) -> None:
            pass

        self.task_mgr.keyboardInterruptHandler = keyboardInterruptHandler

        # Disable the task manager loop for now. Only useful if onscreen.
        self.shutdown()

        # Active enhanced rendering if discrete NVIDIA GPU is used.
        # Note that shadow resolution larger than 1024 significantly affects
        # the frame rate on Intel GPU chipsets: going from 1024 to 2048 makes
        # it drop from 60FPS to 30FPS.
        driver_vendor = self.win.gsg.driver_vendor.lower()
        if any(driver_vendor.startswith(name) for name in ('nvidia', 'apple')):
            self._shadow_size = 4096
        else:
            self._shadow_size = 1024

        # Enable anti-aliasing
        self.render.set_antialias(AntialiasAttrib.MMultisample)

        # Configure lighting and shadows
        self._spotlight = self.config.GetBool('enable-spotlight', False)
        self._lights_mask = [True, True]

        # Create physics-based shader and adapt lighting accordingly.
        # It slows down the rendering by about 30% on discrete NVIDIA GPU.
        shader_options = {'ENABLE_SHADOWS': True}
        pbr_shader = simplepbr.shaderutils.make_shader(
            'pbr', 'simplepbr.vert', 'simplepbr.frag', shader_options)
        self.render.set_attrib(ShaderAttrib.make(pbr_shader))
        env_map = simplepbr.EnvMap.create_empty()
        self.render.set_shader_input(
            'filtered_env_map', env_map.filtered_env_map)
        self.render.set_shader_input(
            'max_reflection_lod',
            env_map.filtered_env_map.num_loadable_ram_mipmap_images)
        self.render.set_shader_input('sh_coeffs', env_map.sh_coefficients)
        self._lights = [
            self._make_light_ambient((0.5, 0.5, 0.5)),
            self._make_light_direct(1, (1.0, 1.0, 1.0), pos=(8.0, -8.0, 10.0))]

        # Define default camera pos
        self._camera_defaults = CAMERA_POS_DEFAULT
        self.reset_camera(*self._camera_defaults)

        # Define clock. It will be used later to limit framerate
        self.clock = ClockObject.get_global_clock()
        self.framerate: Optional[float] = None

        # Create scene tree
        self._scene_root = self.render.attach_new_node('scene_root')
        self._scene_scale = self.config.GetFloat('scene-scale', 1.0)
        self._scene_root.set_scale(self._scene_scale)
        self._groups: Dict[str, NodePath] = {}

        # Create default scene objects
        self._fog = self._make_fog()
        self._axes = self._make_axes()
        self._grid = self._make_grid()
        self._floor = self._make_floor()

        # Create gradient for skybox
        self.skybox = make_gradient_skybox(
            SKY_TOP_COLOR, SKY_BOTTOM_COLOR, 0.35, 0.17)
        self.skybox.set_shader_auto(True)
        self.skybox.set_light_off()
        self.skybox.hide(self.LightMask)

        # The background needs to be parented to an intermediary node to which
        # a compass effect is applied to keep it at the same position as the
        # camera, while being parented to render.
        pivot = self.render.attach_new_node("pivot")
        effect = CompassEffect.make(self.camera, CompassEffect.P_pos)
        pivot.set_effect(effect)
        self.skybox.reparent_to(pivot)

        # The background needs to keep facing the camera a point behind the
        # camera. Note that only its heading should correspond to that of the
        # camera, while the pitch and roll remain unaffected.
        effect = BillboardEffect.make(
            Vec3.up(), False, True, 0.0, NodePath(),
            Point3(0.0, -10.0, 0.0), False)
        self.skybox.set_effect(effect)

        # Create shared 2D renderer to allow display selectively gui elements
        # on offscreen and onscreen window used for capturing frames.
        self.shared_render_2d = NodePath('shared_render_2d')
        self.shared_render_2d.set_depth_test(False)
        self.shared_render_2d.set_depth_write(False)

        # Create dedicated camera 2D for offscreen rendering
        self.offscreen_camera_2d = NodePath(Camera('offscreen_camera2d'))
        lens = OrthographicLens()
        lens.set_film_size(2, 2)
        lens.set_near_far(-1000, 1000)
        self.offscreen_camera_2d.node().set_lens(lens)
        self.offscreen_camera_2d.reparent_to(self.shared_render_2d)

        # Create dedicated aspect2d for offscreen rendering
        self.offAspect2d = self.shared_render_2d.attach_new_node(
            PGTop("offAspect2d"))
        self.offA2dTopLeft = self.offAspect2d.attach_new_node(
            "offA2dTopLeft")
        self.offA2dTopRight = self.offAspect2d.attach_new_node(
            "offA2dTopRight")
        self.offA2dBottomLeft = self.offAspect2d.attach_new_node(
            "offA2dBottomLeft")
        self.offA2dBottomCenter = self.offAspect2d.attach_new_node(
            "offA2dBottomCenter")
        self.offA2dBottomRight = self.offAspect2d.attach_new_node(
            "offA2dBottomRight")
        self.offA2dTopLeft.set_pos(self.a2dLeft, 0, self.a2dTop)
        self.offA2dTopRight.set_pos(self.a2dRight, 0, self.a2dTop)
        self.offA2dBottomLeft.set_pos(self.a2dLeft, 0, self.a2dBottom)
        self.offA2dBottomCenter.set_pos(0, 0, self.a2dBottom)
        self.offA2dBottomRight.set_pos(self.a2dRight, 0, self.a2dBottom)

        # Define widget overlay
        self.offscreen_graphics_lens: Optional[Lens] = None
        self.offscreen_display_region: Optional[DisplayRegion] = None
        self._help_label = None
        self._watermark: Optional[OnscreenImage] = None
        self._legend: Optional[OnscreenImage] = None
        self._clock: Optional[OnscreenText] = None

        # Define input control
        self.key_map = {"mouse1": 0, "mouse2": 0, "mouse3": 0}

        # Define camera control
        self.zoom_rate = 1.03
        self.camera_lookat = np.zeros(3)
        self.longitude_deg = 0.0
        self.latitude_deg = 0.0
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        # Define object/highlighting selector
        self.picker_ray: Optional[CollisionRay] = None
        self.picker_node: Optional[CollisionNode] = None
        self.picked_object: Optional[Tuple[str, str]] = None
        self.click_mouse_x = 0.0
        self.click_mouse_y = 0.0

        # Create resizeable offscreen buffer.
        # Note that a resizable buffer is systematically created, no matter
        # if the main window is an offscreen non-resizable window or an
        # onscreen resizeable graphical window. It avoids having to handle
        # the two cases separately, especially for screenshot resizing and
        # selective overlay information display. However, it affects the
        # performance significantly. At least 20% on discrete NVIDIA GPU and
        # 50% on integrated Intel GPU.
        self._open_offscreen_window(WINDOW_SIZE_DEFAULT)

        # Set default options
        self.enable_lights(True)
        self.enable_shadow(True)
        self.enable_hdr(False)
        self.enable_fog(False)
        self.show_axes(True)
        self.show_grid(False)
        self.show_floor(True)

    def has_gui(self) -> bool:
        """Whether a onscreen graphical window is opened.
        """
        return any(isinstance(win, GraphicsWindow) for win in self.winList)

    def open_window(self) -> None:
        """Open a graphical window, with offscreen buffer attached on it to
        allow for arbitrary size screenshot.
        """
        # Make sure a graphical window is not already open
        if self.has_gui():
            raise RuntimeError("Only one graphical window can be opened.")

        # Replace the original offscreen window by an onscreen one if possible
        is_success = True
        size = self.win.get_size()
        try:
            self.windowType = 'onscreen'
            self.open_main_window(size=size)
        except Exception:   # pylint: disable=broad-except
            is_success = False
            self.windowType = 'offscreen'
            self.open_main_window(size=size)

        if is_success:
            # Setup mouse and keyboard controls for onscreen display
            self._setup_shortcuts()
            self.disableMouse()
            self.accept("wheel_up", self.handle_key, ["wheelup", 1])
            self.accept("wheel_down", self.handle_key, ["wheeldown", 1])
            for i in range(1, 4):
                self.accept(f"mouse{i}", self.handle_key, [f"mouse{i}", 1])
                self.accept(f"mouse{i}-up", self.handle_key, [f"mouse{i}", 0])
            self.task_mgr.add(
                self.move_orbital_camera_task, "move_camera_task", sort=2)

            # Setup object pickler
            self.picker_ray = CollisionRay()
            self.picker_node = CollisionNode('mouse_ray')
            self.picker_node.set_from_collide_mask(
                GeomNode.get_default_collide_mask())
            self.picker_node.addSolid(self.picker_ray)
            self.picker_traverser = CollisionTraverser('traverser')
            picker_np = self.camera.attachNewNode(self.picker_node)
            self.picker_queue = CollisionHandlerQueue()
            self.picker_traverser.addCollider(picker_np, self.picker_queue)

            # Limit framerate to reduce computation cost
            self.set_framerate(PANDA3D_FRAMERATE_MAX)

        # Create resizeable offscreen buffer
        self._open_offscreen_window(size)

        # Throw exception if opening display has failed
        if not is_success:
            raise RuntimeError(
                "Impossible to open graphical window. Make sure display is "
                "available on system.")

        # Enable the task manager
        self.restart()

    def _open_offscreen_window(self,
                               size: Optional[Tuple[int, int]] = None) -> None:
        """Create new completely independent offscreen buffer, rendering the
        same scene than the main window.
        """
        # Handling of default size
        if size is None:
            size = self.win.get_size()

        # Close existing offscreen display if any.
        # Note that one must remove display region associated with shared 2D
        # renderer, otherwise it will be altered when closing current window.
        if self.buff is not None:
            self.buff.remove_display_region(self.offscreen_display_region)
            self.close_window(self.buff, keepCamera=False)

        # Set offscreen buffer frame properties.
        # Note that accumulator bits and back buffers is not supported by
        # resizeable buffers.
        # See https://github.com/panda3d/panda3d/issues/1121
        fbprops = FrameBufferProperties()
        fbprops.set_rgba_bits(8, 8, 8, 0)
        fbprops.set_float_color(False)
        fbprops.set_depth_bits(16)
        fbprops.set_float_depth(True)
        fbprops.set_multisamples(4)

        # Set offscreen buffer windows properties
        winprops = WindowProperties()
        winprops.set_size(*size)

        # Set offscreen buffer flags to enforce resizeable `GraphicsBuffer`
        flags = GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_refuse_parasite
        flags |= GraphicsPipe.BF_resizeable

        # Create new offscreen buffer
        # Note that it is impossible to create resizeable buffer without an
        # already existing host.
        win = self.graphicsEngine.make_output(
            self.pipe, "offscreen_buffer", 0, fbprops, winprops, flags,
            self.win.get_gsg(), self.win)
        if win is None:
            raise RuntimeError("Faulty graphics pipeline of this machine.")
        self.buff = win

        # Append buffer to the list of windows managed by the ShowBase
        self.winList.append(win)

        # Attach a texture as screenshot requires copying GPU data to RAM
        self.buff.add_render_texture(
            Texture(), GraphicsOutput.RTM_triggered_copy_ram)

        # Create 3D camera region for the scene.
        # Set near distance of camera lens to allow seeing model from close.
        self.offscreen_graphics_lens = PerspectiveLens()
        self.offscreen_graphics_lens.set_near(0.1)
        self.make_camera(
            win, camName='offscreen_camera', lens=self.offscreen_graphics_lens)

        # Create 2D display region for widgets
        self.offscreen_display_region = win.makeMonoDisplayRegion()
        self.offscreen_display_region.set_sort(5)
        self.offscreen_display_region.set_camera(self.offscreen_camera_2d)

        # Adjust aspect ratio
        self._adjust_offscreen_window_aspect_ratio()

        # Force rendering the scene to finalize initialization of the GSG
        self.graphics_engine.render_frame()

        # The buffer must be flipped upside-down manually because using the
        # global option `copy-texture-inverted` distorts the shadows of the
        # onscreen window for some reason. Moreover, it must be done after
        # calling `render_frame` at least once.
        self.buff.inverted = True

    def _adjust_offscreen_window_aspect_ratio(self) -> None:
        """Adjust aspect ratio of offscreen window.

        .. note::
            This method is called automatically after resize.
        """
        # Get aspect ratio
        aspect_ratio = self.get_aspect_ratio(self.buff)

        # Adjust 3D rendering aspect ratio
        assert self.offscreen_graphics_lens is not None
        self.offscreen_graphics_lens.set_aspect_ratio(aspect_ratio)

        # Adjust existing anchors for offscreen 2D rendering
        if aspect_ratio < 1:
            # If the window is TALL, lets expand the top and bottom
            self.offAspect2d.set_scale(1.0, aspect_ratio, aspect_ratio)
            a2dTop = 1.0 / aspect_ratio
            a2dBottom = - 1.0 / aspect_ratio
            a2dLeft = -1
            a2dRight = 1.0
        else:
            # If the window is WIDE, lets expand the left and right
            self.offAspect2d.set_scale(1.0/aspect_ratio, 1.0, 1.0)
            a2dTop = 1.0
            a2dBottom = -1.0
            a2dLeft = -aspect_ratio
            a2dRight = aspect_ratio

        self.offA2dTopLeft.set_pos(a2dLeft, 0, a2dTop)
        self.offA2dTopRight.set_pos(a2dRight, 0, a2dTop)
        self.offA2dBottomLeft.set_pos(a2dLeft, 0, a2dBottom)
        self.offA2dBottomRight.set_pos(a2dRight, 0, a2dBottom)

    def getSize(self, win: Optional[Any] = None) -> Tuple[int, int]:
        """Patched to return the size of the window used for capturing frame by
        default, instead of main window.
        """
        if win is None:
            win = self.buff
        return super().getSize(win)

    def getMousePos(self) -> Tuple[float, float]:
        """Get current mouse position if available.

        .. note::
            Can be overloaded to allow for emulated mouse click.
        """

        # Get mouse position if possible:
        try:
            md = self.win.getPointer(0)
            return md.getX(), md.getY()
        except AttributeError:
            return (float("nan"),) * 2

    def handle_key(self, key: str, value: bool) -> None:
        """Input controller handler.
        """
        if key in ["mouse1", "mouse2", "mouse3"]:
            mouseX, mouseY = self.getMousePos()
            if key == "mouse1":
                if value:
                    self.click_mouse_x, self.click_mouse_y = mouseX, mouseY
                elif (self.click_mouse_x == mouseX and
                        self.click_mouse_y == mouseY):
                    # Do not enable clicking on node for Qt widget since
                    # mouse watcher and picker are not properly configured.
                    if self.picker_ray is not None:
                        self.click_on_node()
            self.last_mouse_x, self.last_mouse_y = mouseX, mouseY
            self.key_map[key] = value
        elif key in ["wheelup", "wheeldown"]:
            cam_dir = self.camera_lookat - np.asarray(self.camera.get_pos())
            if key == "wheelup":
                cam_pos = self.camera_lookat - cam_dir / self.zoom_rate
            else:
                cam_pos = self.camera_lookat - cam_dir * self.zoom_rate
            self.camera.set_pos(*cam_pos)

    def click_on_node(self) -> None:
        """Object selector handler.
        """
        # Remove focus of currently selected object
        picked_object_prev = self.picked_object
        if self.picked_object is not None:
            self.highlight_node(*self.picked_object, False)
            self.picked_object = None

        # Select new object if the user actually clicked on a selectable node
        assert self.picker_ray is not None
        object_found = False
        mpos = self.mouseWatcherNode.getMouse()
        self.picker_ray.set_from_lens(self.camNode, mpos.getX(), mpos.getY())
        self.picker_traverser.traverse(self.render)
        for i in range(self.picker_queue.getNumEntries()):
            self.picker_queue.sortEntries()
            picked_node = self.picker_queue.getEntry(i).getIntoNodePath()
            # Do not allow selecting hidden node
            if not picked_node.isHidden():
                node_path = str(picked_node)
                for group_name in self._groups.keys():
                    group_path = f"render/scene_root/{group_name}/"
                    # Only nodes part of user groups can be selected
                    if node_path.startswith(group_path):
                        name = node_path[len(group_path):]
                        if (group_name, name) != picked_object_prev:
                            self.picked_object = (group_name, name)
                        object_found = True
                        break
                if object_found:
                    break

        # Focus on newly selected node
        if self.picked_object is not None:
            self.highlight_node(*self.picked_object, True)

    def move_orbital_camera_task(self,
                                 task: Optional[Any] = None) -> Optional[int]:
        """Custom control of the camera to be more suitable for robotic
        application than the default one.
        """
        # Get mouse position
        x, y = self.getMousePos()

        # Ensure consistent camera pose and lookat
        self.longitude_deg, self.latitude_deg, _ = self.camera.get_hpr()
        cam_pos = np.asarray(self.camera.get_pos())
        cam_dir = self.camera_lookat - cam_pos
        cam_dist = np.linalg.norm(cam_dir)
        longitude = self.longitude_deg * np.pi / 180.0
        latitude = self.latitude_deg * np.pi / 180.0
        cam_dir_n = np.array([-np.sin(longitude)*np.cos(latitude),
                              np.cos(longitude)*np.cos(latitude),
                              np.sin(latitude)])
        self.camera_lookat = cam_pos + cam_dist * cam_dir_n

        if self.key_map["mouse1"]:
            # Update camera rotation
            self.longitude_deg -= (x - self.last_mouse_x) * 0.2
            self.latitude_deg -= (y - self.last_mouse_y) * 0.2

            # Limit angles to [-180;+180] x [-90;+90]
            if self.longitude_deg > 180.0:
                self.longitude_deg = self.longitude_deg - 360.0
            if self.longitude_deg < -180.0:
                self.longitude_deg = self.longitude_deg + 360.0
            if self.latitude_deg > (90.0 - 0.001):
                self.latitude_deg = 90.0 - 0.001
            if self.latitude_deg < (-90.0 + 0.001):
                self.latitude_deg = -90.0 + 0.001

            longitude = self.longitude_deg * np.pi / 180.0
            latitude = self.latitude_deg * np.pi / 180.0
            cam_dir_n = np.array([-np.sin(longitude) * np.cos(latitude),
                                  np.cos(longitude) * np.cos(latitude),
                                  np.sin(latitude)])
            cam_pos = self.camera_lookat - cam_dist * cam_dir_n
            self.camera.set_pos(*cam_pos)
            self.camera.set_hpr(self.longitude_deg, self.latitude_deg, 0)
        if self.key_map["mouse2"]:
            cam_delta = (y - self.last_mouse_y) * 0.02 * cam_dir_n
            self.camera_lookat -= cam_delta
            cam_pos -= cam_delta
            self.camera.set_pos(*cam_pos)
        elif self.key_map["mouse3"]:
            cam_n1 = np.array([np.cos(longitude),
                               np.sin(longitude),
                               0.0])
            cam_n2 = np.array([-np.sin(longitude) * np.sin(latitude),
                               np.cos(longitude) * np.sin(latitude),
                               -np.cos(latitude)])
            pos_shift = ((x - self.last_mouse_x) * cam_n1 +
                         (y - self.last_mouse_y) * cam_n2) * 0.01
            cam_pos -= pos_shift
            self.camera_lookat -= pos_shift
            self.camera.set_pos(*cam_pos)

        # Store latest mouse position for the next frame
        self.last_mouse_x = x
        self.last_mouse_y = y

        # End task
        if task is not None:
            return task.cont
        return None

    def _make_light_ambient(self, color: Tuple3FType) -> NodePath:
        """Patched to fix wrong color alpha.
        """
        node = super()._make_light_ambient(color)
        node.get_node(0).set_color((*color, 1.0))
        return node

    def _make_light_direct(self,
                           index: int,
                           color: Tuple3FType,
                           pos: Tuple3FType,
                           target: Tuple3FType = (0.0, 0.0, 0.0)
                           ) -> NodePath:
        """Patched to fix wrong color alpha.
        """
        light_path = super()._make_light_direct(index, color, pos, target)
        light_path.get_node(0).set_color((*color, 1.0))
        return light_path

    def _make_axes(self) -> NodePath:
        model = GeomNode('axes')
        model.add_geom(geometry.make_axes())
        node = self.render.attach_new_node(model)
        node.set_render_mode_wireframe()
        if self.win.gsg.driver_vendor.startswith('NVIDIA'):
            node.set_render_mode_thickness(4)
        node.set_antialias(AntialiasAttrib.MLine)
        node.set_shader_auto(True)
        node.set_light_off()
        node.hide(self.LightMask)
        node.set_scale(0.3)
        return node

    def _make_floor(self,
                    geom: Optional[Geom] = None,
                    show_meshes: bool = False) -> NodePath:
        model = GeomNode('floor')
        node = self.render.attach_new_node(model)

        if geom is None:
            for xi in range(-10, 10):
                for yi in range(-10, 10):
                    tile = GeomNode(f"tile-{xi}.{yi}")
                    tile.add_geom(geometry.make_plane(size=(1.0, 1.0)))
                    tile_path = node.attach_new_node(tile)
                    tile_path.set_pos((xi + 0.5, yi + 0.5, 0.0))
                    if (xi + yi) % 2:
                        tile_path.set_color((0.95, 0.95, 1.0, 1.0))
                    else:
                        tile_path.set_color((0.14, 0.14, 0.21, 1.0))
        else:
            model.add_geom(geom)
            node.set_color((0.75, 0.75, 0.85, 1.0))
            if show_meshes:
                render_attrib = node.get_state().get_attrib_def(
                    RenderModeAttrib.get_class_slot())
                node.set_attrib(RenderModeAttrib.make(
                    RenderModeAttrib.M_filled_wireframe,
                    0.5,  # thickness
                    render_attrib.perspective,
                    (0.7, 0.7, 0.7, 1.0)  # wireframe_color
                ))

        # Make the floor two-sided to not see through from below
        node.set_two_sided(True)

        # Set material to render shadows if supported
        material = Material()
        material.set_base_color((1.35, 1.35, 1.35, 1.0))
        node.set_material(material, True)

        # Disable light casting
        node.hide(self.LightMask)

        # Adjust frustum of the lights to project shadow over the whole scene
        for light_path in self._lights[1:]:
            bmin, bmax = node.get_tight_bounds(light_path)
            lens = light_path.get_node(0).get_lens()
            lens.set_film_offset((bmin.xz + bmax.xz) * 0.5)
            lens.set_film_size(bmax.xz - bmin.xz)
            lens.set_near_far(bmin.y, bmax.y)

        return node

    def show_floor(self, show: bool) -> None:
        if show:
            self._floor.show()
        else:
            self._floor.hide()

    def update_floor(self,
                     geom: Optional[Geom] = None,
                     show_meshes: bool = False) -> NodePath:
        """Update the floor.

        :param geom: Ground profile as a generic geometry object. If None, then
                     a flat tile ground is rendered.
                     Optional: None by default.
        """
        # Check if floor is currently hidden
        is_hidden = self._floor.isHidden()

        # Remove existing floor and create a new one
        self._floor.remove_node()
        self._floor = self._make_floor(geom, show_meshes)

        # Hide the floor if is was previously hidden
        if is_hidden:
            self._floor.hide()

    def append_group(self,
                     root_path: str,
                     remove_if_exists: bool = True,
                     scale: float = 1.0) -> None:
        """Patched to avoid adding new group if 'remove_if_exists' is false,
        otherwise it will be impossible to access to old ones.
        """
        if not remove_if_exists and root_path in self._groups:
            return
        super().append_group(root_path, remove_if_exists, scale)

    def append_node(self,
                    root_path: str,
                    name: str,
                    node: NodePath,
                    frame: Optional[FrameType] = None) -> None:
        """Patched to make sure node's name is valid and set the color scale.
        """
        if not re.match(r'^[A-Za-z0-9_]+$', name):
            raise RuntimeError(
                "Node's name restricted to case-insensitive ASCII "
                "alphanumeric characters plus underscore.")
        node.set_color_scale((*(3 * (1.2,)), 1.0))
        super().append_node(root_path, name, node, frame)

    def highlight_node(self, root_path: str, name: str, enable: bool) -> None:
        """Enable or disable highlighting of a given node.

        .. note::
            It displays the wireframe of the corresponding geometry on top of
            the original shader rendering.

        :param root_path: Path to the root node of the group.
        :param name: Name of the node within the specified group.
        :param enable: Whether hightlighting must be enabled or disabled.
        """
        node = self._groups[root_path].find(name)
        if node:
            render_mode = node.get_render_mode()
            if enable:
                if render_mode == RenderModeAttrib.M_filled_wireframe:
                    return
                render_attrib = node.get_state().get_attrib_def(
                    RenderModeAttrib.get_class_slot())
                node.set_attrib(RenderModeAttrib.make(
                    RenderModeAttrib.M_filled_wireframe,
                    1.0,  # thickness (1.0 by default)
                    render_attrib.perspective,
                    (2.0, 2.0, 2.0, 1.0)  # wireframe_color
                ))
            else:
                if render_mode == RenderModeAttrib.M_off:
                    return
                node.clear_render_mode()

    def append_frame(self,
                     root_path: str,
                     name: str,
                     frame: Optional[FrameType] = None) -> None:
        """Append a cartesian frame primitive node to the group.
        """
        model = GeomNode('axes')
        model.add_geom(geometry.make_axes())
        node = NodePath(model)
        node.set_render_mode_wireframe()
        if self.win.gsg.driver_vendor.startswith('NVIDIA'):
            node.set_render_mode_thickness(4)
        node.set_antialias(AntialiasAttrib.MLine)
        node.set_shader_auto(True)
        node.set_light_off()
        node.hide(self.LightMask)
        self.append_node(root_path, name, node, frame)

    def append_cone(self,
                    root_path: str,
                    name: str,
                    radius: float,
                    length: float,
                    num_sides: int = 16,
                    frame: Optional[FrameType] = None) -> None:
        """Append a cone primitive node to the group.
        """
        geom_node = GeomNode("cone")
        geom_node.add_geom(make_cone(num_sides))
        node = NodePath(geom_node)
        node.set_scale(radius, radius, length)
        self.append_node(root_path, name, node, frame)

    def append_torus(self,
                     root_path: str,
                     name: str,
                     major_radius: float,
                     minor_radius: float = 0.2,
                     num_sides: int = 12,
                     frame: Optional[FrameType] = None) -> None:
        """Append a torus primitive node to the group.
        """
        geom_node = GeomNode("torus")
        geom_node.add_geom(make_torus(minor_radius / major_radius, num_sides))
        node = NodePath(geom_node)
        node.set_scale(major_radius, major_radius, major_radius)
        self.append_node(root_path, name, node, frame)

    def append_cylinder(self,  # pylint: disable=arguments-renamed
                        root_path: str,
                        name: str,
                        radius: float,
                        length: float,
                        theta_start: float = 0.0,
                        theta_end: float = 2.0 * math.pi,
                        anchor_bottom: bool = False,
                        frame: Optional[FrameType] = None) -> None:
        """Patched to add optional to place anchor at the bottom of the
        cylinder instead of the middle.
        """
        geom_node = GeomNode('cylinder')
        geom_node.add_geom(make_pie(theta_start, theta_end))
        node = NodePath(geom_node)
        node.set_scale(Vec3(radius, radius, length))
        if anchor_bottom:
            node.set_pos(0.0, 0.0, length / 2.0)
        self.append_node(root_path, name, node, frame)

    def append_arrow(self,
                     root_path: str,
                     name: str,
                     radius: float,
                     length: float,
                     anchor_top: bool = False,
                     frame: Optional[FrameType] = None) -> None:
        """Append an arrow primitive node to the group.

        ..note::
            The arrow is aligned with z-axis in world frame, and the tip is at
            position (0.0, 0.0, 0.0) in world frame.
        """
        arrow_geom = GeomNode("arrow")
        arrow_node = NodePath(arrow_geom)
        head = make_cone()
        head_geom = GeomNode("head")
        head_geom.add_geom(head)
        head_node = NodePath(head_geom)
        head_node.reparent_to(arrow_node.attach_new_node("head"))
        head_node.set_scale(1.75, 1.75, 3.5 * radius)
        head_node.set_pos(0.0, 0.0, length)
        body = geometry.make_cylinder()
        body_geom = GeomNode("body")
        body_geom.add_geom(body)
        body_node = NodePath(body_geom)
        body_node.reparent_to(arrow_node.attach_new_node("body"))
        body_node.set_scale(1.0, 1.0, length)
        body_node.set_pos(0.0, 0.0, (-0.5 if anchor_top else 0.5) * length)
        arrow_node.set_scale(radius, radius, 1.0)
        self.append_node(root_path, name, arrow_node, frame)

    def append_mesh(self,
                    root_path: str,
                    name: str,
                    mesh_path: str,
                    scale: Optional[Tuple3FType] = None,
                    frame: Optional[FrameType] = None,
                    no_cache: Optional[bool] = None) -> None:
        """Append a mesh node to the group.

        :param root_path: Path to the group's root node
        :param name: Node name within a group
        :param mesh_path: Path to the mesh file on disk
        :param scale: Mesh scale.
                    Optional: No rescaling by default.
        :param frame: Local frame position and quaternion.
                      Optional: ((0., 0., 0.), (0., 0., 0., 0.)) by default.
        :param no_cache: Use cache to load a model.
                         Optional: may depend on the mesh file.
        """
        # Load the mesh
        mesh = self.loader.loadModel(mesh_path, noCache=no_cache)

        # Fix the orientation of the mesh if it has '.dae' extension
        if mesh_path.lower().endswith('.dae'):
            # Replace non-standard hard drive prefix on Windows
            if sys.platform.startswith('win'):
                mesh_path = re.sub(
                    r'^/([A-Za-z])', lambda m: m[1].upper() + ":", mesh_path)

            # Parse the mesh file toe extract axis up if provided
            def parse_xml(xml_path: str) -> Tuple[ET.Element, Dict[str, str]]:
                xml_iter = ET.iterparse(xml_path, events=["start-ns"])
                xml_namespaces = dict(prefix_namespace_pair
                                      for _, prefix_namespace_pair in xml_iter)
                return (xml_iter.root,  # type: ignore[attr-defined]
                        xml_namespaces)

            root, ns = parse_xml(mesh_path)
            if ns:
                field_axis = root.find(f".//{{{ns['']}}}up_axis")
            else:
                field_axis = root.find(".//up_axis")

            # Change the orientation of the mesh if necessary
            if field_axis is not None:
                assert field_axis.text is not None
                axis = field_axis.text.lower()
                if axis == 'z_up':
                    mesh.set_mat(Mat4.yToZUpMat())

        # Set the scale of the mesh
        if scale is not None:
            mesh.set_scale(*scale)

        # Render meshes two-sided in panda3d to avoid seeing through it
        mesh.set_two_sided(True)

        self.append_node(root_path, name, mesh, frame)

    def set_watermark(self,
                      img_fullpath: Optional[str] = None,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> None:
        """Add an image on the bottom left corner of the window, as part of
        the 2D overlay. It will always appear on foreground.

        :param img_fullpath: Full path of the image. Internally, the format
                             must be supported by `panda3d.core.OnscreenImage`.
                             Note that alpha (transparency) is supported.
        :param width: Desired absolute width of the image in pixels. The aspect
                      ratio is not preserved if inconsistent with the original
                      width and height.
        :param height: Desired absolute height of the image in pixels.
        """
        # Remove existing watermark, if any
        if self._watermark is not None:
            self._watermark.remove_node()
            self._watermark = None

        # Do nothing if img_fullpath is not specified
        if img_fullpath is None or img_fullpath == "":
            return

        # Get image size if not user-specified
        if width is None or height is None:
            image_header = PNMImageHeader()
            image_header.readHeader(Filename(img_fullpath))
            width = width or int(image_header.getXSize())
            height = height or int(image_header.getYSize())

        # Compute relative image size
        width_win, height_win = self.getSize()
        width_rel, height_rel = width / width_win, height / height_win

        # Make sure it does not take too much space of window
        if width_rel > WATERMARK_SCALE_MAX:
            width_rel = WATERMARK_SCALE_MAX
            height_rel = WATERMARK_SCALE_MAX * height_rel / width_rel
        if height_rel > WATERMARK_SCALE_MAX:
            height_rel = WATERMARK_SCALE_MAX
            width_rel = WATERMARK_SCALE_MAX * width_rel / height_rel

        # Create image watermark on main window
        self._watermark = OnscreenImage(image=img_fullpath,
                                        parent=self.a2dBottomLeft,
                                        scale=(width_rel, 1, height_rel))
        self._watermark.set_transparency(TransparencyAttrib.MAlpha)

        # Add it on secondary window
        self.offA2dBottomLeft.node().add_child(self._watermark.node())

        # Move the watermark in bottom left corner
        self._watermark.set_pos(
            WIDGET_MARGIN_REL + width_rel, 0, WIDGET_MARGIN_REL + height_rel)

        # Flip the vertical axis
        assert self.buff is not None
        if self.buff.inverted:
            self._watermark.set_tex_scale(TextureStage.getDefault(), 1.0, -1.0)

    def set_legend(self,
                   items: Optional[Sequence[
                       Tuple[str, Optional[Sequence[int]]]]] = None) -> None:
        """Add a matplotlib legend on bottom center on the window, as part of
        the 2D overlay. It will always appear on foreground.

        .. warning::
            This method requires installing `matplolib` manually since it is an
            optional dependency.

        :param items: Sequence of pair (label, color), where `label` is a
                      string and `color` is a sequence of integer (R, G, B
                      [, A]). `None` to disable.
                      Optional: `None` by default.
        """
        # Make sure plot submodule is available
        try:
            # pylint: disable=import-outside-toplevel
            from matplotlib import cbook
            from matplotlib.patches import Patch
        except ImportError:
            warnings.warn(
                "Method not supported. Please install 'jiminy_py[plot]'.",
                category=UserWarning, stacklevel=2)
            return

        # Remove existing legend, if any
        if self._legend is not None:
            self.offA2dBottomCenter.node().remove_child(self._legend.node())
            self._legend.remove_node()
            self._legend = None

        # Do nothing if items is not specified
        if items is None or not items:
            return

        # Create non-interactive headless figure unrelated to current backend
        width_win, height_win = self.getSize()
        plt_agg = importlib.import_module(cbook._backend_module_name('Agg'))
        manager = plt_agg.new_figure_manager(
            num=0, figsize=(width_win / LEGEND_DPI, height_win / LEGEND_DPI),
            dpi=LEGEND_DPI)
        fig = manager.canvas.figure
        ax = fig.subplots()

        # Render the legend
        color_default = (0.0, 0.0, 0.0, 1.0)
        handles = [Patch(color=c or color_default, label=t) for t, c in items]
        legend = ax.legend(handles=handles,
                           ncol=len(handles),
                           framealpha=1,
                           frameon=True)
        ax.set_axis_off()
        fig.draw(renderer=fig.canvas.get_renderer())

        # Compute bbox size to be power of 2 for software rendering.
        bbox = legend.get_window_extent().padded(2)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox_pixels = LEGEND_DPI * np.array(bbox_inches.extents)
        bbox_size_2 = (2 ** np.ceil(
            np.log2(bbox_pixels[2:] - bbox_pixels[:2]))).astype(dtype=int)
        bbox_size_delta = bbox_size_2 - (bbox_pixels[2:] - bbox_pixels[:2])
        bbox_pixels[:2] = np.floor(bbox_pixels[:2] - 0.5 * bbox_size_delta)
        bbox_pixels[2:] = bbox_pixels[:2] + bbox_size_2 + 0.1
        bbox_inches = bbox.from_extents(bbox_pixels / LEGEND_DPI)

        # Export the figure, limiting the bounding box to the legend area,
        # slightly extended to ensure the surrounding rounded corner box of
        # is not cropped. Transparency is enabled, so it is not an issue.
        io_buf = io.BytesIO()
        fig.savefig(
            io_buf, format='rgba', dpi='figure', transparent=True,
            bbox_inches=bbox_inches)
        io_buf.seek(0)
        img_raw = io_buf.getvalue()

        # Delete the legend along with its temporary figure
        manager.destroy()

        # Create texture in which to render the image buffer
        tex = Texture()
        tex.setup2dTexture(
            *bbox_size_2, Texture.T_unsigned_byte, Texture.F_rgba8)
        tex.set_ram_image_as(img_raw, 'rgba')

        # Compute relative image size, ignoring the real width of the
        # texture since it has transparent background.
        # We assume that the width of the window is the limiting dimension.
        width = int(bbox_pixels[2] - bbox_pixels[0])
        legend_scale_rel = min(
            (1.0 - 2 * WIDGET_MARGIN_REL) * width_win / width,
            LEGEND_SCALE_MAX) * width / width_win
        width_rel = legend_scale_rel * (tex.x_size / width)
        height_rel = width_rel * (tex.y_size / tex.x_size)

        # Create legend on main window
        self._legend = OnscreenImage(image=tex,
                                     parent=self.a2dBottomCenter,
                                     scale=(width_rel, 1, height_rel))
        self._legend.set_transparency(TransparencyAttrib.MAlpha)

        # Add legend on offscreen window
        self.offA2dBottomCenter.node().add_child(self._legend.node())

        # Move the legend in top left corner
        self._legend.set_pos(0, 0, WIDGET_MARGIN_REL + height_rel)

        # Flip the vertical axis
        assert self.buff is not None
        if self.buff.inverted:
            self._legend.set_tex_scale(TextureStage.getDefault(), 1.0, -1.0)

    def set_clock(self, time: Optional[float] = None) -> None:
        """Add a clock on the bottom right corner of the window, as part of
        the 2D overlay. It will always appear on foreground.

        .. warning::
            This method requires installing `matplolib` manually since it is an
            optional dependency.

        :param time: Current time in seconds as a float. Its fractional part
                     can be used to specified milliseconds but anything smaller
                     will be ignored.
        """
        # Make sure plot submodule is available
        try:
            # pylint: disable=import-outside-toplevel
            from matplotlib import font_manager
        except ImportError as e:
            raise ImportError(
                "Method not available. Please install 'jiminy_py[plot]'."
                ) from e

        # Remove existing watermark, if any
        if time is None:
            if self._clock is not None:
                self._clock.remove_node()
                self._clock = None
            return

        if self._clock is None:
            # Get path of default matplotlib font
            fontpath = _sanitize_path(font_manager.findfont(None))

            # Create clock on main window.
            self._clock = OnscreenText(
                text="00:00:00.000",
                parent=self.a2dBottomRight,
                scale=CLOCK_SCALE,
                font=self.loader.loadFont(fontpath),
                fg=(1, 0, 0, 1),
                bg=(1, 1, 1, 1),
                frame=(0, 0, 0, 1),
                mayChange=True,
                align=TextNode.ARight)

            # Add it on secondary window
            self.offA2dBottomRight.node().add_child(self._clock.node())

            # Fix card margins not uniform
            self._clock.textNode.set_card_as_margin(0.2, 0.2, 0.05, 0)
            self._clock.textNode.set_frame_as_margin(0.2, 0.2, 0.05, 0)

            # Move the clock in bottom right corner
            card_dims = self._clock.textNode.get_card_transformed()
            self._clock.set_pos(-WIDGET_MARGIN_REL-card_dims[1],
                                0,
                                WIDGET_MARGIN_REL-card_dims[2])

        # Update clock values
        hours, remainder = divmod(time, 3600)
        minutes, seconds = divmod(remainder, 60)
        remainder, seconds = math.modf(seconds)
        milliseconds = 1000 * remainder
        self._clock.setText(f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"
                            f".{milliseconds:03.0f}")

    def set_material(self,
                     root_path: str,
                     name: str,
                     color: Optional[Tuple4FType] = None,
                     texture_path: str = '',
                     disable_material: Optional[bool] = None) -> None:
        """Patched to avoid raising an exception if node does not exist, and to
        clear color if not specified. In addition, texture are disabled if the
        color is specified, and a physics-based shader is used if available.
        """
        # Handling of default argument
        if disable_material is None:
            disable_material = color is not None

        node = self._groups[root_path].find(name)
        if node:
            if disable_material:
                node.set_texture_off(1)
            else:
                node.clear_texture()
                node.clear_material()

            if color is None:
                node.clear_color()
            else:
                node.set_color(Vec4(*color))
                node.set_color_scale(
                    4 * (1.0,) if texture_path else (*(3 * (1.2,)), 1.0))

                material = Material()
                material.set_ambient(Vec4(*color))
                material.set_diffuse(Vec4(*color))
                material.set_specular(Vec3(1, 1, 1))
                material.set_roughness(0.4)
                node.set_material(material, True)

                if color[3] < 1.0:
                    node.set_transparency(TransparencyAttrib.M_alpha)
                else:
                    node.set_transparency(TransparencyAttrib.M_none)

            if texture_path:
                texture = self.loader.load_texture(texture_path)
                node.set_texture(texture)
                node.set_transparency(TransparencyAttrib.M_alpha)

    def set_scale(self,
                  root_path: str,
                  name: str,
                  scale: Tuple3FType) -> None:
        """Override scale of node of a node.
        """
        node = self._groups[root_path].find(name)
        if node:
            if any(abs(s) < 1e-3 for s in scale):
                if not node.is_hidden():
                    node.set_tag("status", "disable")
                    node.hide()
            else:
                node.set_scale(*scale)
                if node.get_tag("status") == "disable":
                    node.set_tag("status", "auto")
                    node.show()

    def set_scales(self,
                   root_path: str,
                   name_scales_dict: Dict[str, Tuple3FType]) -> None:
        """Override scale of nodes within a group.
        """
        for name, scale in name_scales_dict.items():
            self.set_scale(root_path, name, scale)

    def move_node(self,
                  root_path: str,
                  name: str,
                  frame: FrameType) -> None:
        """Set pose of a single node.
        """
        node = self._groups[root_path].find(name)
        if isinstance(frame, np.ndarray):
            node.set_mat(Mat4(*frame.T.flat))
        else:
            pos, quat = frame
            node.set_pos_quat(Vec3(*pos), Quat(*quat))

    def remove_node(self, root_path: str, name: str) -> None:
        """Remove a single node from the scene.
        """
        node = self._groups[root_path].find(name)
        if node:
            node.remove_node()

    def show_node(self,
                  root_path: str,
                  name: str,
                  show: bool,
                  always_foreground: Optional[bool] = None) -> None:
        """Turn rendering on or off for a single node.
        """
        node = self._groups[root_path].find(name)
        if node:
            if show:
                if node.get_tag("status") in ("hidden", ""):
                    node.set_tag("status", "auto")
                    node.show()
            else:
                node.set_tag("status", "hidden")
                node.hide()
            if always_foreground is not None:
                if always_foreground:
                    node.set_bin("fixed", 0)
                else:
                    node.clear_bin()
                node.set_depth_test(not always_foreground)
                node.set_depth_write(not always_foreground)

    def get_camera_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current absolute pose of the camera.

        :returns: Gather the current position (X, Y, Z) and quaternion
                  representation of the orientation (X, Y, Z, W) as a
                  pair of `np.ndarray`.
        """
        return (np.array(self.camera.get_pos()),
                np.array(self.camera.get_quat()))

    def set_camera_transform(self,
                             pos: Tuple3FType,
                             quat: np.ndarray,
                             lookat: Tuple3FType = (0.0, 0.0, 0.0)) -> None:
        """Set the current absolute pose of the camera.

        :param pos: Desired position of the camera.
        :param quat: Desired orientation of the camera as a quaternion
                     (X, Y, Z, W).
        :param lookat: Point at which the camera is looking at. It is partially
                       redundant with the desired orientation and will take
                       precedence in case of inconsistency. It is also involved
                       in zoom control.
        """
        self.camera.set_pos(*pos)
        self.camera.set_quat(LQuaternion(quat[-1], *quat[:-1]))
        self.camera_lookat = np.array(lookat)
        self.move_orbital_camera_task()

    def get_camera_lookat(self) -> np.ndarray:
        """Get the location of the point toward which the camera is looking at.
        """
        return self.camera_lookat

    def set_camera_lookat(self,
                          pos: Tuple3FType) -> None:
        """Set the point at which the camera is looking at.

        :param pos: Position of the point at which the camera is looking at.
        """
        self.camera.set_pos(
            self.camera.get_pos() + Vec3(*pos) - Vec3(*self.camera_lookat))
        self.camera_lookat = np.asarray(pos)

    def set_window_size(self, width: int, height: int) -> None:
        """Set the size of the offscreen window used for screenshot.

        :param width: Width of the window in pixels.
        :param height: Height of the window in pixels.
        """
        assert self.buff is not None
        self.buff.set_size(width, height)
        self._adjust_offscreen_window_aspect_ratio()

    def set_framerate(self, framerate: Optional[float] = None) -> None:
        """Limit framerate of Panda3d to avoid consuming too much ressources.

        :param framerate: Desired framerate limit. None to disable.
                          Optional: Disable framerate limit by default.
        """
        if framerate is not None:
            self.clock.set_mode(ClockObject.MLimited)
            self.clock.set_frame_rate(PANDA3D_FRAMERATE_MAX)
        else:
            self.clock.set_mode(ClockObject.MNormal)
        self.framerate = framerate

    def get_framerate(self) -> Optional[float]:
        """Get current framerate limit.
        """
        return self.framerate

    def save_screenshot(self, filename: Optional[str] = None) -> bool:
        """Save a screenshot of the scene from the current viewpoint of the
        camera as an image file.

        :param filename: Fullpath where store the image of the local
                         filesystem. An extension must be specified to indicate
                         the desired format. The later must be supported by
                         `panda3d.core.PNMImage`.
                         Optional: Store a PNG image in the working directory
                         named 'screenshot-%Y-%m-%d-%H-%M-%S.png' by default.

        :returns: Whether export has been successful.
        """
        # Generate filename based on current time if not provided
        if filename is None:
            template = 'screenshot-%Y-%m-%d-%H-%M-%S.png'
            filename = datetime.now().strftime(template)

        # Refresh the scene to make sure it is perfectly up-to-date.
        # It will take into account the updated position of the camera.
        assert self.buff is not None
        self.buff.trigger_copy()
        self.graphics_engine.render_frame()

        # Capture frame as image
        image = PNMImage()
        if not self.buff.get_screenshot(image):
            return False

        # Flip the image if the buffer is also flipped to revert the effect*
        if self.buff.inverted:
            image.flip(flip_x=False, flip_y=True, transpose=False)

        # Remove alpha if format does not support it
        if not filename.lower().endswith('.png'):
            image.remove_alpha()

        # Save the image
        if not image.write(filename):
            return False

        return True

    def get_screenshot(self,
                       requested_format: str = 'RGB',
                       raw: bool = False) -> Union[np.ndarray, bytes]:
        """Patched to take screenshot of the last window available instead of
        the main one, and to add raw data return mode for efficient
        multiprocessing.

        .. warning::
            Note that the speed of this method is limited by the global
            framerate, as any other method relaying on low-level panda3d task
            scheduler. The framerate limit must be disable manually to avoid
            such limitation.

        .. note::
            Internally, Panda3d uses BGRA, so using it is slightly faster than
            RGBA, but not RGB since there is one channel missing.

        :param requested_format: Desired export format (e.g. 'RGB' or 'BGRA')
        :param raw: whether to return a raw memory view of bytes, of a
                    structured `np.ndarray` of uint8 with dimensions [H, W, D].
        """
        # Refresh the scene
        assert self.buff is not None
        self.buff.trigger_copy()
        self.graphics_engine.render_frame()

        # Get frame as raw texture
        assert self.buff is not None
        texture = self.buff.get_texture()

        # Extract raw array buffer from texture
        image = texture.get_ram_image_as(requested_format)

        # Return raw buffer if requested
        if raw:
            return image.get_data()

        # Convert raw texture to numpy array if requested
        xsize, ysize = texture.get_x_size(), texture.get_y_size()
        return np.frombuffer(image, np.uint8).reshape((ysize, xsize, -1))

    def enable_shadow(self, enable: bool) -> None:
        """Enable or disable shadow casting.

        :param enable: Whether shoadows must be enabled or disabled.
        """
        for light in self._lights:
            if not light.node().is_ambient_light():
                light.node().set_shadow_caster(enable)
        self.render.set_depth_offset(-2 if enable else 0)
        self._shadow_enabled = enable


class Panda3dProxy(mp.Process):
    """Panda3d viewer instance running in separate process for asynchronous
    execution.

    Based on `panda3d_viewer.ViewerAppProxy`:
    https://github.com/ikalevatykh/panda3d_viewer/blob/1105e082b75943aa0a81e623e003d1d649c85a14/panda3d_viewer/viewer_proxy.py
    """  # noqa: E501  # pylint: disable=line-too-long
    @wraps(Panda3dApp.__init__)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Start an application in a sub-process.
        """
        super().__init__()
        self._args = args
        self._kwargs = kwargs
        self._is_async = False
        self.daemon = True
        self._host_conn, self._proc_conn = mp.Pipe()
        self.start()
        reply = self._host_conn.recv()
        if isinstance(reply, Exception):
            raise reply

    def __getstate__(self) -> Dict[str, Any]:
        """Required for Windows and OS X support, which use spawning instead of
        forking to create subprocesses, requiring pickling objects.
        """
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Defined for the same reason than `__getstate__`.
        """
        self.__dict__.update(state)

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), dir(Panda3dApp))

    def async_mode(self) -> AbstractContextManager:
        """Context specifically designed for executing methods asynchronously.

        .. note::
            Using this mode would result in a significant speed up if waiting
            for request completion is not necessary, tpyically when refreshing
            the pose of nodes, adding some geometries, removing others...etc

        .. warning::
            Beware that methods called asynchronously always returns `None`.
            Their original outputs are simply discarded if any. Moreover, an
            exception raised during asynchronous exception will only be printed
            right before the next method execution instead of being thrown on
            the spot.
        """
        proxy_ref = ref(self)

        class ContextAsyncMode(AbstractContextManager):
            """Context manager forcing async execution when forwarding request
            to the underlying panda3d viewer instance.
            """
            def __enter__(self) -> None:
                nonlocal proxy_ref
                proxy = proxy_ref()
                assert proxy is not None
                proxy._is_async = True

            def __exit__(self,
                         exc_type: Optional[Type[BaseException]],
                         exc_value: Optional[BaseException],
                         traceback: Optional[TracebackType]) -> None:
                nonlocal proxy_ref
                proxy = proxy_ref()
                assert proxy is not None
                proxy._is_async = False

        return ContextAsyncMode()

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Forward method and attribute lookup to the underlying panda3d viewer
        instance.

        :param name: Name of the instance method or attribute.

        :returns: Callable that would redirect the attribute access or instance
                  method call. It ensures signature matching and it preserves
                  the original docstring.
        """
        @wraps(getattr(Panda3dApp, name))
        def _send(*args: Any, **kwargs: Any) -> Any:
            if self._host_conn.closed:
                raise ViewerClosedError("Viewer not available anymore.")
            while self._host_conn.poll():
                try:
                    reply = self._host_conn.recv()
                except EOFError:
                    pass
                if isinstance(reply, Exception):
                    if isinstance(reply, ViewerClosedError):
                        # Close pipe to make sure it is not used in future
                        self._host_conn.close()
                        raise reply
                    traceback = TracebackException.from_exception(reply)
                    print(''.join(traceback.format()))
            self._host_conn.send((name, args, kwargs, self._is_async))
            if self._is_async:
                return None
            if self._host_conn.poll(PANDA3D_REQUEST_TIMEOUT):
                reply = self._host_conn.recv()
            else:
                # Something is wrong... aborting to prevent potential deadlock
                self._host_conn.send(("stop", (), (), True))
                self._host_conn.close()
                raise ViewerClosedError(
                    "Viewer has been because it did not respond.")
            if isinstance(reply, Exception):
                if isinstance(reply, ViewerClosedError):
                    # Close pipe to make sure it is not used in future
                    self._host_conn.close()
                raise reply
            return reply

        return _send

    def run(self) -> None:
        """Run a panda3d viewer instance in a sub-process.
        """
        # pylint: disable=broad-exception-caught
        try:
            app = Panda3dApp(*self._args, **self._kwargs)
            self._proc_conn.send(None)

            def _execute(task: Task) -> Optional[int]:
                # pylint: disable=unused-argument
                for _ in range(100):
                    if self._proc_conn.closed:
                        return Task.done
                    if not self._proc_conn.poll(0.0002):
                        break
                    name, args, kwargs, is_async = self._proc_conn.recv()
                    try:
                        reply = getattr(app, name)(*args, **kwargs)
                    except Exception as error:
                        reply = error
                        if is_async:
                            raise
                    if not is_async:
                        self._proc_conn.send(reply)
                return Task.cont

            app.task_mgr.add(_execute, "Communication task", -50)
            app.run()
        except Exception as error:
            self._proc_conn.send(error)
        else:
            self._proc_conn.send(
                ViewerClosedError('User closed the main window'))
        # read the rest to prevent the host process from being blocked
        if self._proc_conn.poll(0.05):
            self._proc_conn.recv()
        self._proc_conn.close()


class Panda3dViewer:
    """A Panda3D based viewer.
    """
    def __init__(self,
                 window_title: str = 'jiminy',
                 window_type: Literal['onscreen', 'offscreen'] = 'onscreen',
                 config: Optional[Union[Dict[str, Any], ViewerConfig]] = None,
                 **kwargs: Any) -> None:
        """Open a window, setup a scene.

        :param window_title: Title of the window for onscreen rendering.
                             Optional: 'onscreen' by default.
        :param window_type: Whether the window is rendered 'onscreen' or
                            'offscreen'. Beware onscreen rendering requieres a
                            graphics server, eg Wayland or X11 on Linux.
                            Optional: 'onscreen' by default.
        :param config: Viewer options forwarded at instantiation.
                       Optional: None by default.
        """
        if config is None:
            config = ViewerConfig()
        elif isinstance(config, dict):
            config = ViewerConfig(**config)
        assert isinstance(config, ViewerConfig)
        for key, value in kwargs.items():
            config.set_value(key, value)
        config.set_window_title(window_title)
        config.set_value('window-type', window_type)
        self._window_type = window_type

        if window_type == 'onscreen':
            # run application asynchronously in a sub-process
            self._app = Panda3dProxy(config)
        elif window_type == 'offscreen':
            # start application in the main process
            self._app = Panda3dApp(config)
        else:
            raise ViewerError(f"Unknown window type: '{window_type}'")

    def join(self) -> None:
        """Run the application until the user close the main window.
        """
        if self._window_type == 'onscreen':
            self._app.join()

    def stop(self) -> None:
        """Stop the application.
        """
        try:
            self._app.stop()
        except ViewerError:
            return
        self.destroy()

    def destroy(self) -> None:
        """Destroy the application and free all resources.
        """
        if self._window_type == 'offscreen':
            self._app.destroy()

    def __getattr__(self, name: str) -> Any:
        """Forward methods to the  underlying panda3d viewer instance.

        :param name: Name of the instance method or attribute.
        """
        return getattr(self.__getattribute__('_app'), name)

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), dir(self._app))


def convert_bvh_collision_geometry_to_primitive(geom: hppfcl.CollisionGeometry
                                                ) -> Optional[Geom]:
    """Convert a triangle-based collision geometry associated to a primitive
    geometry for rendering it with Panda3D.

    :param geom: Collision geometry to convert.
    """
    # Extract vertices and faces from geometry
    vertices, faces = extract_vertices_and_faces_from_geometry(geom)

    # Return immediately if there is nothing to load
    if len(faces) == 0:
        return None

    # Define normal to vertices as the average normal of adjacent triangles
    fnormals = np.cross(vertices[faces[:, 2]] - vertices[faces[:, 1]],
                        vertices[faces[:, 0]] - vertices[faces[:, 1]])
    fnormals /= np.linalg.norm(fnormals, axis=0)
    normals = np.zeros((len(vertices), 3), dtype=np.float32)
    for i in range(3):
        normals[faces[:, i]] += fnormals
    normals /= np.linalg.norm(normals, axis=0)

    # Create primitive triangle geometry
    vformat = GeomVertexFormat()
    vformat.addArray(GeomVertexArrayFormat(
        "vertex", 3, Geom.NT_float32, Geom.C_point))
    vformat.addArray(GeomVertexArrayFormat(
        "normal", 3, Geom.NT_float32, Geom.C_normal))
    vformat = GeomVertexFormat.registerFormat(vformat)
    vdata = GeomVertexData('vdata', vformat, Geom.UHStatic)
    vdata.modify_array_handle(0).copy_data_from(vertices)
    vdata.modify_array_handle(1).copy_data_from(normals)
    prim = GeomTriangles(Geom.UHStatic)
    prim.set_index_type(GeomEnums.NTUint32)
    prim.modify_vertices(-1).modify_handle().copy_data_from(faces)

    # Create geometry object
    geom = Geom(vdata)
    geom.add_primitive(prim)

    return geom


class Panda3dVisualizer(BaseVisualizer):
    """A Pinocchio display using panda3d engine.

    Based on https://github.com/stack-of-tasks/pinocchio/blob/master/bindings/python/pinocchio/visualize/panda3d_visualizer.py
    Copyright (c) 2014-2020, CNRS
    Copyright (c) 2018-2020, INRIA
    """  # noqa: E501  # pylint: disable=line-too-long
    def initViewer(self,  # pylint: disable=arguments-differ
                   viewer: Optional[Union[Panda3dViewer, Panda3dApp]] = None,
                   loadModel: bool = False,
                   **kwargs: Any) -> None:
        """Init the viewer by attaching to / creating a GUI viewer.
        """
        self.visual_group: Optional[str] = None
        self.collision_group: Optional[str] = None
        self.display_visuals = False
        self.display_collisions = False
        self.viewer = viewer

        if viewer is None:
            self.viewer = Panda3dViewer(window_title="jiminy")

        if loadModel:
            self.loadViewerModel(root_node_name=self.model.name)

    def getViewerNodeName(self,
                          geometry_object: pin.GeometryObject,
                          geometry_type: pin.GeometryType) -> Tuple[str, str]:
        """Return the name of the geometry object inside the viewer.
        """
        if geometry_type is pin.GeometryType.VISUAL:
            assert self.visual_group is not None
            return self.visual_group, geometry_object.name
        # if geometry_type is pin.GeometryType.COLLISION:
        assert self.collision_group is not None
        return self.collision_group, geometry_object.name

    def loadViewerGeometryObject(self,
                                 geometry_object: pin.GeometryObject,
                                 geometry_type: pin.GeometryType,
                                 color: Optional[np.ndarray] = None) -> None:
        """Load a single geometry object
        """
        # Assert(s) for type checker
        assert self.viewer is not None

        # Skip ground plane
        if geometry_object.name == "ground":
            return

        # Get node name
        node_name = self.getViewerNodeName(geometry_object, geometry_type)

        # Extract geometry information
        geom = geometry_object.geometry
        mesh_path = geometry_object.meshPath
        texture_path = ""
        if geometry_object.overrideMaterial:
            # Get material from URDF. The color is only used if no texture or
            # if its value is not the default because meshColor is never unset.
            if os.path.exists(geometry_object.meshTexturePath):
                texture_path = geometry_object.meshTexturePath
            if color is None and (not texture_path or any(
                    geometry_object.meshColor != [0.9, 0.9, 0.9, 1.0])):
                color = geometry_object.meshColor

        # Try to load mesh from path first, to take advantage of very effective
        # Panda3d model caching procedure.
        if os.path.exists(mesh_path):
            # append a mesh
            mesh_path = _sanitize_path(geometry_object.meshPath)
            scale = npToTuple(geometry_object.meshScale)
            self.viewer.append_mesh(*node_name, mesh_path, scale)
        else:
            # Each geometry must have at least a color or a texture
            if color is None and not texture_path:
                color = np.array([0.75, 0.75, 0.75, 1.0])

            # Append a primitive geometry
            if isinstance(geom, hppfcl.Capsule):
                self.viewer.append_capsule(
                    *node_name, geom.radius, 2.0 * geom.halfLength)
            elif isinstance(geom, hppfcl.Cylinder):
                self.viewer.append_cylinder(
                    *node_name, geom.radius, 2.0 * geom.halfLength)
            elif isinstance(geom, hppfcl.Cone):
                self.viewer.append_cone(
                    *node_name, geom.radius, 2.0 * geom.halfLength)
            elif isinstance(geom, hppfcl.Box):
                self.viewer.append_box(*node_name, 2.0 * geom.halfSide)
            elif isinstance(geom, hppfcl.Sphere):
                self.viewer.append_sphere(*node_name, geom.radius)
            else:
                # Create primitive triangle geometry
                try:
                    obj = convert_bvh_collision_geometry_to_primitive(geom)
                except ValueError:
                    warnings.warn(
                        "Unsupported geometry type for "
                        f"{geometry_object.name} ({type(geom)})",
                        category=UserWarning, stacklevel=2)
                    return

                # Add the primitive geometry to the scene
                geom_node = GeomNode(geometry_object.name)
                geom_node.add_geom(obj)
                node = NodePath(geom_node)
                node.set_two_sided(True)
                self.viewer.append_node(*node_name, node)

        # Set material
        self.viewer.set_material(*node_name, color, texture_path)

    def loadViewerModel(self,  # pylint: disable=arguments-differ
                        root_node_name: str,
                        color: Optional[np.ndarray] = None) -> None:
        """Create a group of nodes displaying the robot meshes in the viewer.
        """
        # Assert(s) for type checker
        assert self.viewer is not None

        self.root_name = root_node_name

        # Load robot visual meshes
        self.visual_group = "/".join((self.root_name, "visuals"))
        self.viewer.append_group(self.visual_group)
        for visual in self.visual_model.geometryObjects:
            self.loadViewerGeometryObject(
                visual, pin.GeometryType.VISUAL, color)
        self.displayVisuals(True)

        # Load robot collision meshes
        self.collision_group = "/".join((self.root_name, "collisions"))
        self.viewer.append_group(self.collision_group)
        for collision in self.collision_model.geometryObjects:
            self.loadViewerGeometryObject(
                collision, pin.GeometryType.COLLISION, color)
        self.displayCollisions(False)

    def display(self,  # pylint: disable=signature-differs
                q: np.ndarray) -> None:
        """Display the robot at configuration q in the viewer by placing all
        the bodies."""
        pin.forwardKinematics(self.model, self.data, q)

        def move(group: str, model: pin.Model, data: pin.Data) -> None:
            # Assert(s) for type checker
            assert self.viewer is not None

            pin.updateGeometryPlacements(self.model, self.data, model, data)
            name_pose_dict = {}
            for obj in model.geometryObjects:
                oMg = data.oMg[model.getGeometryId(obj.name)]
                x, y, z, qx, qy, qz, qw = pin.SE3ToXYZQUAT(oMg)
                name_pose_dict[obj.name] = ((x, y, z), (qw, qx, qy, qz))
            self.viewer.move_nodes(group, name_pose_dict)

        if self.display_visuals:
            assert self.visual_group is not None
            move(self.visual_group, self.visual_model, self.visual_data)

        if self.display_collisions:
            assert self.collision_group is not None
            move(self.collision_group, self.collision_model,
                 self.collision_data)

    def displayCollisions(self, visibility: bool) -> None:
        """Set whether to display collision objects or not."""
        # Assert(s) for type checker
        assert self.viewer is not None and self.collision_group is not None

        self.viewer.show_group(self.collision_group, visibility)
        self.display_collisions = visibility

    def displayVisuals(self, visibility: bool) -> None:
        """Set whether to display visual objects or not."""
        # Assert(s) for type checker
        assert self.viewer is not None and self.visual_group is not None

        self.viewer.show_group(self.visual_group, visibility)
        self.display_visuals = visibility
