import io
import os
import re
import sys
import math
import array
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import PureWindowsPath
from typing import Callable, Optional, Dict, Tuple, Union, Sequence, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

from panda3d.core import (
    NodePath, Point3, Vec3, Mat4, Quat, LQuaternion, Geom, GeomEnums, GeomNode,
    GeomVertexData, GeomTriangles, GeomVertexArrayFormat, GeomVertexFormat,
    GeomVertexWriter, CullFaceAttrib, GraphicsWindow, PNMImage, InternalName,
    OmniBoundingVolume, CompassEffect, BillboardEffect, Filename, TextNode,
    Texture, TextureStage, PNMImageHeader, PGTop, Camera, PerspectiveLens,
    TransparencyAttrib, OrthographicLens, ClockObject, GraphicsPipe,
    WindowProperties, FrameBufferProperties, loadPrcFileData, AntialiasAttrib)
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText

import panda3d_viewer
import panda3d_viewer.viewer_app
import panda3d_viewer.viewer_proxy
from panda3d_viewer import geometry
from panda3d_viewer import (Viewer as Panda3dViewer,
                            ViewerConfig as Panda3dViewerConfig)
from panda3d_viewer.viewer_errors import ViewerClosedError

import hppfcl
import pinocchio as pin
from pinocchio.utils import npToTuple
from pinocchio.visualize import BaseVisualizer


WINDOW_SIZE_DEFAULT = (500, 500)
CAMERA_POS_DEFAULT = [(4.0, -4.0, 1.5), (0, 0, 0.5)]

LEGEND_DPI = 400
LEGEND_SCALE = 0.3
CLOCK_SCALE = 0.1
WIDGET_MARGIN_REL = 0.05

PANDA3D_FRAMERATE_MAX = 30


Tuple3FType = Union[Tuple[float, float, float], np.ndarray]
Tuple4FType = Union[Tuple[float, float, float, float], np.ndarray]
FrameType = Union[Tuple[Tuple3FType, Tuple4FType], np.ndarray]


def make_gradient_skybox(sky_color: Tuple3FType,
                         ground_color: Tuple3FType,
                         offset: float = 0.0,
                         subdiv: int = 2):
    """Simple gradient to be used as skybox.

    For reference, see:
    - https://discourse.panda3d.org/t/color-gradient-scene-background/26946/14
    """
    # Check validity of arguments
    assert subdiv >= 2, "Number of sub-division must be larger than 2."
    assert 0.0 <= offset and offset <= 1.0, "Offset must be in [0.0, 1.0]."

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
    # straight up.
    vertex_data = GeomVertexData(
        "prism_data", vformat, GeomEnums.UH_static)
    vertex_data.unclean_set_num_rows(4 + subdiv * 2)
    values = array.array("f", (-1000., -50., 86.6, 1000., -50., 86.6))
    offset_angle = np.pi / 1.5 * offset
    delta_angle = (np.pi / .75 - offset_angle * 2.) / (subdiv + 1)
    for i in range(subdiv):
        angle = np.pi / 3. + offset_angle + delta_angle * (i + 1)
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
    prism.set_light_off(1)
    prism.set_bin("background", 0)
    prism.set_depth_write(False)
    prism.set_depth_test(False)

    return prism


def make_cone(num_sides: int = 16) -> Geom:
    """Create a close shaped cone, approximate by a pyramid with regular
    convex n-sided polygon base.

    For reference about refular polygon:
    https://en.wikipedia.org/wiki/Regular_polygon
    """
    # Define vertex format
    vformat = GeomVertexFormat.get_v3n3t2()
    vdata = GeomVertexData('vdata', vformat, Geom.UH_static)
    vdata.uncleanSetNumRows(num_sides + 2)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    tcoord = GeomVertexWriter(vdata, 'texcoord')

    # Add radial points
    for u in np.linspace(0.0, 2 * np.pi, num_sides):
        x, y = math.cos(u), math.sin(u)
        vertex.addData3(x, y, 0.0)
        normal.addData3(x, y, 0.0)
        tcoord.addData2(x, y)

    # Add top and bottom points
    vertex.addData3(0.0, 0.0, 1.0)
    normal.addData3(0.0, 0.0, 1.0)
    tcoord.addData2(0.0, 0.0)
    vertex.addData3(0.0, 0.0, 0.0)
    normal.addData3(0.0, 0.0, -1.0)
    tcoord.addData2(0.0, 0.0)

    # Make triangles.
    # Note that by default, rendering is one-sided. It only renders the outside
    # face, that is defined based on the "winding" order of the vertices making
    # the triangles. For reference, see:
    # https://discourse.panda3d.org/t/procedurally-generated-geometry-and-the-default-normals/24986/2
    prim = GeomTriangles(Geom.UH_static)
    for i in range(num_sides - 1):
        prim.addVertices(i, i + 1, num_sides)
        prim.addVertices(i + 1, i, num_sides + 1)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


class Panda3dApp(panda3d_viewer.viewer_app.ViewerApp):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Enforce viewer configuration
        config = Panda3dViewerConfig()
        config.set_window_size(*WINDOW_SIZE_DEFAULT)
        config.set_window_fixed(False)
        config.enable_antialiasing(True, multisamples=2)
        config.set_value('framebuffer-software', '0')
        config.set_value('framebuffer-hardware', '0')
        config.set_value('load-display', 'pandagl')
        config.set_value('aux-display',
                         'p3headlessgl'
                         '\naux-display pandadx9'
                         '\naux-display pandadx8'
                         '\naux-display p3tinydisplay')
        config.set_value('window-type', 'offscreen')
        config.set_value('default-near', 0.1)
        config.set_value('assimp-optimize-graph', True)
        config.set_value('gl-version', '3 1')
        config.set_value('notify-level', 'error')
        config.set_value('notify-level-x11display', 'fatal')
        config.set_value('default-directnotify-level', 'error')
        loadPrcFileData('', str(config))

        # Define offscreen buffer
        self.buff = None

        # Initialize base implementation.
        # Note that the original constructor is by-passed on purpose.
        ShowBase.__init__(self)

        # Configure rendering
        self.render.set_shader_auto()
        self.render.set_antialias(AntialiasAttrib.MAuto)

        # Define default camera pos
        self._camera_defaults = CAMERA_POS_DEFAULT
        self.reset_camera(*self._camera_defaults)

        # Define clock. It will be used later to limit framerate.
        self.clock = ClockObject.getGlobalClock()
        self.framerate = None

        # Configure lighting and shadows
        # Note that shadow resolution larger than 1024 significatly affects
        # the frame rate on Intel GPU chipsets: going from 1024 to 2048 makes
        # it drop from 60FPS to 30FPS.
        self._spotlight = self.config.GetBool('enable-spotlight', False)
        self._shadow_size = self.config.GetInt('shadow-buffer-size', 1024)
        self._lights = [self._make_light_ambient((0.5, 0.5, 0.5)),
                        self._make_light_direct(
                            1, (0.5, 0.5, 0.5), pos=(8.0, 8.0, 10.0))]
        self._lights_mask = [True, True]
        self.enable_lights(True)

        # Create default scene objects
        self._fog = self._make_fog()
        self._axes = self._make_axes()
        self._grid = self._make_grid()
        self._floor = self._make_floor()

        # Create scene tree
        self._scene_root = self.render.attach_new_node('scene_root')
        self._scene_scale = self.config.GetFloat('scene-scale', 1.0)
        self._scene_root.set_scale(self._scene_scale)
        self._groups = {}

        # Create gradient for skybox
        sky_color = (0.53, 0.8, 0.98, 1.0)
        ground_color = (0.1, 0.1, 0.43, 1.0)
        self.skybox = make_gradient_skybox(sky_color, ground_color, 0.7)

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
        self.sharedRender2d = NodePath('sharedRender2d')
        self.sharedRender2d.setDepthTest(False)
        self.sharedRender2d.setDepthWrite(False)

        # Create dedicated camera 2D for offscreen rendering
        self.offCamera2d = NodePath(Camera('off_camera2d'))
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setNearFar(-1000, 1000)
        self.offCamera2d.node().setLens(lens)
        self.offCamera2d.reparentTo(self.sharedRender2d)

        # Create dedicated aspect2d for offscreen rendering
        self.offAspect2d = self.sharedRender2d.attach_new_node(
            PGTop("offAspect2d"))
        self.offA2dTopLeft = self.offAspect2d.attach_new_node(
            "offA2dTopLeft")
        self.offA2dTopRight = self.offAspect2d.attach_new_node(
            "offA2dTopRight")
        self.offA2dBottomLeft = self.offAspect2d.attach_new_node(
            "offA2dBottomLeft")
        self.offA2dBottomRight = self.offAspect2d.attach_new_node(
            "offA2dBottomRight")
        self.offA2dTopLeft.set_pos(self.a2dLeft, 0, self.a2dTop)
        self.offA2dTopRight.set_pos(self.a2dRight, 0, self.a2dTop)
        self.offA2dBottomLeft.set_pos(self.a2dLeft, 0, self.a2dBottom)
        self.offA2dBottomRight.set_pos(self.a2dRight, 0, self.a2dBottom)

        # Initialize onscreen display and controls internal state
        self._help_label = None
        self._watermark = None
        self._legend = None
        self._clock = None
        self.offGraphicsLens = None
        self.offDisplayRegion = None
        self.zoom_rate = 1.03
        self.camera_lookat = np.zeros(3)
        self.key_map = {"mouse1": 0, "mouse2": 0, "mouse3": 0}
        self.longitudeDeg = 0.0
        self.latitudeDeg = 0.0

        # Create resizeable offscreen buffer
        self._open_offscreen_window(WINDOW_SIZE_DEFAULT)

        # Set default options
        self.enable_lights(True)
        self.enable_shadow(True)
        self.enable_hdr(False)
        self.enable_fog(False)
        self.show_axes(True)
        self.show_grid(False)
        self.show_floor(True)

    def open_window(self) -> None:
        # Make sure a graphical window is not already open
        if any(isinstance(win, GraphicsWindow) for win in self.winList):
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
            self.taskMgr.add(
                self.move_orbital_camera_task, "move_camera_task", sort=2)

            # Limit framerate to reduce computation cost
            self.set_framerate(PANDA3D_FRAMERATE_MAX)

        # Create resizeable offscreen buffer
        self._open_offscreen_window(size)

        # Throw exception if opening display has failed
        if not is_success:
            raise RuntimeError(
                "Impossible to open graphical window. Make sure display is "
                "available on system.")

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
            self.buff.remove_display_region(self.offDisplayRegion)
            self.close_window(self.buff, keepCamera=False)

        # Set offscreen buffer frame properties
        # Note that accumalator bits and back buffers is not supported by
        # resizeable buffers.
        fbprops = FrameBufferProperties(self.win.getFbProperties())
        fbprops.set_accum_bits(0)
        fbprops.set_back_buffers(0)

        # Set offscreen buffer windows properties
        winprops = WindowProperties()
        winprops.set_size(*size)

        # Set offscreen buffer flags to enforce resizeable `GaphicsBuffer`
        flags = GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_refuse_parasite
        flags |= GraphicsPipe.BF_resizeable

        # Create new offscreen buffer.
        # Note that it is impossible to create resizeable buffer without an
        # already existing host for some reason...
        win = self.graphicsEngine.make_output(
            self.pipe, "off_buffer", 0, fbprops, winprops, flags,
            self.win.get_gsg(), self.win)

        # Append buffer to the list of windows managed by the ShowBase
        self.buff = win
        self.winList.append(win)

        # Create 3D camera region for the scene.
        # Set near distance of camera lens to allow seeing model from close.
        self.offGraphicsLens = PerspectiveLens()
        self.offGraphicsLens.set_near(0.1)
        self.make_camera(win, camName='off_camera', lens=self.offGraphicsLens)

        # Create 2D display region for widgets
        self.offDisplayRegion = win.makeMonoDisplayRegion()
        self.offDisplayRegion.set_sort(5)
        self.offDisplayRegion.set_camera(self.offCamera2d)

        # # Adjust aspect ratio
        self._adjust_offscreen_window_aspect_ratio()

    def _adjust_offscreen_window_aspect_ratio(self) -> None:
        # Get aspect ratio
        aspect_ratio = self.get_aspect_ratio(self.buff)

        # Adjust 3D rendering aspect ratio
        self.offGraphicsLens.set_aspect_ratio(aspect_ratio)

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
            self.offAspect2d.setScale(1.0 / aspect_ratio, 1.0, 1.0)
            a2dTop = 1.0
            a2dBottom = -1.0
            a2dLeft = -aspect_ratio
            a2dRight = aspect_ratio

        self.offA2dTopLeft.set_pos(a2dLeft, 0, a2dTop)
        self.offA2dTopRight.set_pos(a2dRight, 0, a2dTop)
        self.offA2dBottomLeft.set_pos(a2dLeft, 0, a2dBottom)
        self.offA2dBottomRight.set_pos(a2dRight, 0, a2dBottom)

    def getSize(self, win: Optional[Any] = None) -> Tuple[int, int]:
        """Must be patched to return the size of the window used for capturing
        frame by default, instead of main window.
        """
        if win is None:
            win = self.buff
        return super().getSize(win)

    def getMousePos(self) -> Tuple[int, int]:
        md = self.win.getPointer(0)
        return md.getX(), md.getY()

    def handle_key(self, key: str, value: bool) -> None:
        if key in ["mouse1", "mouse2", "mouse3"]:
            self.lastMouseX, self.lastMouseY = self.getMousePos()
            self.key_map[key] = value
        elif key in ["wheelup", "wheeldown"]:
            cam_dir = self.camera_lookat - np.asarray(self.camera.get_pos())
            if key == "wheelup":
                cam_pos = self.camera_lookat - cam_dir / self.zoom_rate
            else:
                cam_pos = self.camera_lookat - cam_dir * self.zoom_rate
            self.camera.set_pos(*cam_pos)

    def move_orbital_camera_task(self, task: Any) -> None:
        # Get mouse position
        x, y = self.getMousePos()

        # Ensure consistent camera pose and lookat
        self.longitudeDeg, self.latitudeDeg, _ = self.camera.get_hpr()
        cam_pos = np.asarray(self.camera.get_pos())
        cam_dir = self.camera_lookat - cam_pos
        cam_dist = np.linalg.norm(cam_dir)
        longitudeRad = self.longitudeDeg * np.pi / 180.0
        latitudeRad = self.latitudeDeg * np.pi / 180.0
        cam_dir_n = np.array([-np.sin(longitudeRad)*np.cos(latitudeRad),
                              np.cos(longitudeRad)*np.cos(latitudeRad),
                              np.sin(latitudeRad)])
        self.camera_lookat = cam_pos + cam_dist * cam_dir_n

        if self.key_map["mouse1"]:
            # Update camera rotation
            self.longitudeDeg -= (x - self.lastMouseX) * 0.2
            self.latitudeDeg -= (y - self.lastMouseY) * 0.2

            # Limit angles to [-180;+180] x [-90;+90]
            if (self.longitudeDeg > 180.0):
                self.longitudeDeg = self.longitudeDeg - 360.0
            if (self.longitudeDeg < -180.0):
                self.longitudeDeg = self.longitudeDeg + 360.0
            if (self.latitudeDeg > (90.0 - 0.001)):
                self.latitudeDeg = 90.0 - 0.001
            if (self.latitudeDeg < (-90.0 + 0.001)):
                self.latitudeDeg = -90.0 + 0.001

            longitudeRad = self.longitudeDeg * np.pi / 180.0
            latitudeRad = self.latitudeDeg * np.pi / 180.0
            cam_dir_n = np.array([-np.sin(longitudeRad) * np.cos(latitudeRad),
                                  np.cos(longitudeRad) * np.cos(latitudeRad),
                                  np.sin(latitudeRad)])
            cam_pos = self.camera_lookat - cam_dist * cam_dir_n
            self.camera.set_pos(*cam_pos)
            self.camera.set_hpr(self.longitudeDeg, self.latitudeDeg, 0)
        if self.key_map["mouse2"]:
            cam_delta = (y - self.lastMouseY) * 0.02 * cam_dir_n
            self.camera_lookat -= cam_delta
            cam_pos -= cam_delta
            self.camera.set_pos(*cam_pos)
        elif self.key_map["mouse3"]:
            cam_n1 = np.array([np.cos(longitudeRad),
                               np.sin(longitudeRad),
                               0.0])
            cam_n2 = np.array([-np.sin(longitudeRad) * np.sin(latitudeRad),
                               np.cos(longitudeRad) * np.sin(latitudeRad),
                               -np.cos(latitudeRad)])
            pos_shift = ((x - self.lastMouseX) * cam_n1 +
                         (y - self.lastMouseY) * cam_n2) * 0.01
            cam_pos -= pos_shift
            self.camera_lookat -= pos_shift
            self.camera.set_pos(*cam_pos)

        # Store latest mouse position for the next frame
        self.lastMouseX = x
        self.lastMouseY = y

        # End task
        return task.cont

    def _make_light_ambient(self, color: Tuple3FType) -> NodePath:
        """Must be patched to fix wrong color alpha.
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
        """Must be patched to fix wrong color alpha.
        """
        node = super()._make_light_direct(index, color, pos, target)
        node.get_node(0).set_color((*color, 1.0))
        return node

    def _make_axes(self) -> NodePath:
        node = super()._make_axes()
        node.set_scale(0.33)
        return node

    def _make_floor(self) -> NodePath:
        model = GeomNode('floor')
        node = self.render.attach_new_node(model)
        for xi in range(-10, 11):
            for yi in range(-10, 11):
                tile = GeomNode(f"tile-{xi}.{yi}")
                tile.add_geom(geometry.make_plane(size=(1.0, 1.0)))
                tile_path = node.attach_new_node(tile)
                tile_path.set_pos((xi, yi, 0.0))
                if (xi + yi) % 2:
                    tile_path.set_color((0.95, 0.95, 1.0, 1))
                else:
                    tile_path.set_color((0.13, 0.13, 0.2, 1))
        node.set_two_sided(True)
        return node

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

    def append_arrow(self,
                     root_path: str,
                     name: str,
                     radius: float,
                     length: float,
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
        head_geom.addGeom(head)
        head_node = NodePath(head_geom)
        head_node.reparent_to(arrow_node.attach_new_node("head"))
        head_node.set_scale(1.75, 1.75, 3.5*radius)
        head_node.set_pos(0.0, 0.0, -3.5*radius)
        body = geometry.make_cylinder()
        body_geom = GeomNode("body")
        body_geom.addGeom(body)
        body_node = NodePath(body_geom)
        body_node.reparent_to(arrow_node.attach_new_node("body"))
        body_node.set_scale(1.0, 1.0, length)
        body_node.set_pos(0.0, 0.0, -length/2-3.5*radius)
        arrow_node.set_scale(radius, radius, 1.0)
        self.append_node(root_path, name, arrow_node, frame)

    def append_mesh(self,
                    root_path: str,
                    name: str,
                    mesh_path: str,
                    scale: Optional[Tuple3FType] = None,
                    frame: Union[np.ndarray, Tuple[
                        Union[np.ndarray, Sequence[float]],
                        Union[np.ndarray, Sequence[float]]]] = None,
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
        mesh = self.loader.loadModel(mesh_path, noCache=no_cache)
        if mesh_path.lower().endswith('.dae'):
            def parse_xml(xml_path: str) -> Tuple[ET.Element, Dict[str, str]]:
                xml_iter = ET.iterparse(xml_path, events=["start-ns"])
                xml_namespaces = dict(prefix_namespace_pair
                                      for _, prefix_namespace_pair in xml_iter)
                return xml_iter.root, xml_namespaces

            # Replace non-standard hard drive prefix on Windows
            if sys.platform.startswith('win'):
                mesh_path = re.sub(r'^/([A-Za-z])',
                                   lambda m: m.group(1).upper() + ":",
                                   mesh_path)

            root, ns = parse_xml(mesh_path)
            if ns:
                field_axis = root.find(f".//{{{ns['']}}}up_axis")
            else:
                field_axis = root.find(".//up_axis")
            if field_axis is not None:
                axis = field_axis.text.lower()
                if axis == 'z_up':
                    mesh.set_mat(Mat4.yToZUpMat())
        if scale is not None:
            mesh.set_scale(*scale)
            if sum([s < 0 for s in scale]) % 2 != 0:
                # reverse the cull order in case of negative scale values
                mesh.set_attrib(CullFaceAttrib.make_reverse())
        self.append_node(root_path, name, mesh, frame)

    def set_watermark(self,
                      img_fullpath: Optional[str] = None,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> None:
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
            width = width or float(image_header.getXSize())
            height = height or float(image_header.getYSize())

        # Compute relative image size
        width_win, height_win = self.getSize()
        width_rel, height_rel = width / width_win, height / height_win

        # Make sure it does not take too much space of window
        if width_rel > 0.2:
            width_rel, height_rel = 0.2, height_rel / width_rel * 0.2
        if height_rel > 0.2:
            width_rel, height_rel = width_rel / height_rel * 0.2, 0.2

        # Create image watermark on main window
        self._watermark = OnscreenImage(image=img_fullpath,
                                        parent=self.a2dBottomLeft,
                                        scale=(width_rel, 1, height_rel))

        # Add it on secondary window
        self.offA2dBottomLeft.node().add_child(self._watermark.node())

        # Move the watermark in bottom right corner
        self._watermark.set_pos(
            WIDGET_MARGIN_REL + width_rel, 0, WIDGET_MARGIN_REL + height_rel)

        # Refresh frame
        self.step()

    def set_legend(self,
                   items: Optional[Dict[str, Optional[Sequence[int]]]] = None
                   ) -> None:
        # Remove existing watermark, if any
        if self._legend is not None:
            self._legend.remove_node()
            self._legend = None

        # Do nothing if items is not specified
        if items is None or not items:
            return

        # Create empty figure with the legend
        color_default = np.array([0.0, 0.0, 0.0, 1.0])
        handles = [Patch(color=c if c is not None else color_default, label=t)
                   for t, c in items.items()]
        fig = plt.figure()
        legend = fig.gca().legend(handles=handles, framealpha=1, frameon=True)
        fig.gca().set_axis_off()

        # Render the legend
        fig.canvas.draw()

        # Compute bbox size to be power of 2 for software rendering.
        bbox = legend.get_window_extent().padded(2)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox_pixels = np.array(bbox_inches.extents) * LEGEND_DPI
        bbox_pixels = np.floor(bbox_pixels)
        bbox_pixels[2:] = bbox_pixels[:2] + 2 ** np.ceil(np.log(
            bbox_pixels[2:] - bbox_pixels[:2]) / np.log(2.0)) + 0.1
        bbox_inches = bbox.from_extents(bbox_pixels / LEGEND_DPI)

        # Export the figure, limiting the bounding box to the legend area,
        # slighly extended to ensure the surrounding rounded corner box of
        # is not cropped. Transparency is enabled, so it is not an issue.
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='rgba', dpi=LEGEND_DPI, transparent=True,
                    bbox_inches=bbox_inches)
        io_buf.seek(0)
        img_raw = io_buf.getvalue()
        width, height = map(int, bbox_pixels[2:] - bbox_pixels[:2])

        # Delete the legend along with its temporary figure
        plt.close(fig)

        # Create texture in which to render the image buffer
        tex = Texture()
        tex.setup2dTexture(
            width, height, Texture.T_unsigned_byte, Texture.F_rgba8)
        tex.set_ram_image(img_raw)

        # Compute relative image size
        width_win, height_win = self.getSize()
        width_rel = LEGEND_SCALE * width / width_win
        height_rel = LEGEND_SCALE * height / height_win

        # Create legend on main window
        self._legend = OnscreenImage(image=tex,
                                     parent=self.a2dTopLeft,
                                     scale=(width_rel, 1, height_rel))

        # Add it on secondary window
        self.offA2dTopLeft.node().add_child(self._legend.node())

        # Move the legend in top left corner
        self._legend.set_pos(
            WIDGET_MARGIN_REL + width_rel, 0, - WIDGET_MARGIN_REL - height_rel)

        # Flip the vertical axis and enable transparency
        self._legend.set_transparency(TransparencyAttrib.MAlpha)
        self._legend.set_tex_scale(TextureStage.getDefault(), 1.0, -1.0)

        # Refresh frame
        self.step()

    def set_clock(self, time: Optional[float] = None) -> None:
        # Remove existing watermark, if any
        if time is None:
            if self._clock is not None:
                self._clock.remove_node()
                self._clock = None
            return

        if self._clock is None:
            # Create clock on main window.
            # Note that the default matplotlib font will be used.
            self._clock = OnscreenText(
                text="00:00:00.000",
                parent=self.a2dBottomRight,
                scale=CLOCK_SCALE,
                font=self.loader.loadFont(font_manager.findfont(None)),
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
                                WIDGET_MARGIN_REL-card_dims[2])

        # Update clock values
        hours, remainder = divmod(time, 3600)
        minutes, seconds = divmod(remainder, 60)
        remainder, seconds = math.modf(seconds)
        milliseconds = 1000 * remainder
        self._clock.setText(f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"
                            f".{milliseconds:03.0f}")

        # Refresh frame
        self.step()

    def set_material(self,
                     root_path: str,
                     name: str,
                     color: Optional[Tuple4FType] = None,
                     texture_path: str = '',
                     disable_material: bool = False) -> None:
        """Must be patched to avoid raising an exception if node does not
        exist, and to clear color if not specified. In addition, an optional
        argument to disable texture and has been added.
        """
        node = self._groups[root_path].find(name)
        if node:
            if disable_material:
                node.set_texture_off(1)
            else:
                node.clear_texture()
                node.clear_material()
            super().set_material(root_path, name, color, texture_path)
            if color is None:
                node.clear_color()

    def move_node(self,
                  root_path: str,
                  name: str,
                  frame: FrameType) -> None:
        """Set pose of a single node.
        """
        node = self._groups[root_path].find(name).children[0]
        if isinstance(frame, np.ndarray):
            node.set_mat(Mat4(*frame.T.flat))
        else:
            pos, quat = frame
            node.set_pos_quat(Vec3(*pos), Quat(*quat))

    def set_scale(self,
                  root_path: str,
                  name: str,
                  scale: Optional[Tuple3FType] = None) -> None:
        """Override scale of node of a node.
        """
        node = self._groups[root_path].find(name).children[0]
        node.set_scale(*scale)

    def set_scales(self, root_path, name_scales_dict):
        """Override scale of nodes within a group.
        """
        for name, scale in name_scales_dict.items():
            self.set_scale(root_path, name, scale)

    def remove_node(self, root_path: str, name: str) -> None:
        """Remove a signle node from the scene.
        """
        node = self._groups[root_path].find(name)
        if node:
            node.remove_node()

    def set_camera_transform(self,
                             pos: Tuple3FType,
                             quat: np.ndarray) -> None:
        self.camera.set_pos(*pos)
        self.camera.set_quat(LQuaternion(quat[-1], *quat[:-1]))
        self.camera_lookat = np.zeros(3)
        self.step()  # Update frame on-the-spot

    def set_window_size(self, width: int, height: int) -> None:
        self.buff.setSize(width, height)
        self._adjust_offscreen_window_aspect_ratio()
        self.step()  # Update frame on-the-spot

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
        if filename is None:
            template = 'screenshot-%Y-%m-%d-%H-%M-%S.png'
            filename = datetime.now().strftime(template)
        image = PNMImage()
        if not self.buff.get_screenshot(image):
            return False
        if not filename.lower().endswith('.png'):
            image.remove_alpha()
        if not image.write(filename):
            return False
        return True

    def get_screenshot(self,
                       requested_format: str = 'RGBA',
                       raw: bool = False) -> np.ndarray:
        """Must be patched to take screenshot of the last window available
        instead of the main one, and to add raw data return mode for efficient
        multiprocessing.

        .. warning::
            Note that the speed of this method is limited by the global
            framerate, as any other method relaying on low-level panda3d task
            scheduler. The framerate limit must be disable manually to avoid
            such limitation.
        """
        # Capture frame as raw texture
        texture = self.buff.get_screenshot()

        # Extract raw array buffer from texture
        image = texture.get_ram_image_as(requested_format)

        # Return raw texture if requested
        if raw:
            return image.get_data()

        # Convert raw texture to numpy array if requested
        xsize = texture.get_x_size()
        ysize = texture.get_y_size()
        dsize = len(requested_format)
        array = np.frombuffer(image, np.uint8).reshape((ysize, xsize, dsize))
        return np.flipud(array)

    def enable_shadow(self, enable: bool) -> None:
        for light in self._lights:
            if not light.node().is_ambient_light():
                light.node().set_shadow_caster(enable)
        self.render.set_depth_offset(-1 if enable else 0)
        self._shadow_enabled = enable


class Panda3dProxy(panda3d_viewer.viewer_proxy.ViewerAppProxy):
    def __getstate__(self) -> dict:
        """Required for Windows support, which uses spawning instead of forking
        to create subprocesses, requiring pickling of process instance.
        """
        return vars(self)

    def __setstate__(self, state: dict) -> None:
        """Must be defined for the same reason than `__getstate__`.
        """
        vars(self).update(state)

    def __getattr__(self, name: str) -> Callable:
        """Must be overloaded to catch closed window to avoid deadlock.
        """
        def _send(*args, **kwargs):
            if self._host_conn.closed:
                raise ViewerClosedError('User closed the main window')
            self._host_conn.send((name, args, kwargs))
            reply = self._host_conn.recv()
            if isinstance(reply, Exception):
                if isinstance(reply, ViewerClosedError):
                    # Close pipe to make sure it does not get used in future
                    self._host_conn.close()
                raise reply
            return reply

        return _send

    def run(self) -> None:
        """Must be patched to use Jiminy ViewerApp instead of the original one.
        """
        panda3d_viewer.viewer_app.ViewerApp = Panda3dApp  # noqa
        return super().run()

panda3d_viewer.viewer_proxy.ViewerAppProxy = Panda3dProxy  # noqa


def delegate(self, name: str) -> Any:
    return getattr(self.__getattribute__('_app'), name)


Panda3dViewer.__getattr__ = delegate
delattr(Panda3dViewer, 'set_material')


class Panda3dVisualizer(BaseVisualizer):
    """A Pinocchio display using panda3d engine.

    Based on https://github.com/stack-of-tasks/pinocchio/blob/master/bindings/python/pinocchio/visualize/panda3d_visualizer.py
    Copyright (c) 2014-2020, CNRS
    Copyright (c) 2018-2020, INRIA
    """  # noqa: E501
    def initViewer(self,
                   viewer: Optional[Panda3dViewer] = None,
                   loadModel: bool = False) -> None:
        """Init the viewer by attaching to / creating a GUI viewer.
        """
        self.visual_group = None
        self.collision_group = None
        self.display_visuals = False
        self.display_collisions = False
        self.viewer = viewer

        if viewer is None:
            self.viewer = Panda3dViewer(window_title="jiminy")

        if loadModel:
            self.loadViewerModel(rootNodeName=self.model.name)

    def getViewerNodeName(self,
                          geometry_object: hppfcl.CollisionGeometry,
                          geometry_type: pin.GeometryType) -> Tuple[str, str]:
        """Return the name of the geometry object inside the viewer.
        """
        if geometry_type is pin.GeometryType.VISUAL:
            return self.visual_group, geometry_object.name
        elif geometry_type is pin.GeometryType.COLLISION:
            return self.collision_group, geometry_object.name

    def loadViewerGeometryObject(self,
                                 geometry_object: hppfcl.CollisionGeometry,
                                 geometry_type: pin.GeometryType,
                                 color: Optional[np.ndarray] = None) -> None:
        """Load a single geometry object
        """
        # Skip ground plane
        if geometry_object.name == "ground":
            return

        # Get node name
        node_name = self.getViewerNodeName(geometry_object, geometry_type)

        # Create panda3d object based on the geometry and add it to the scene
        geom = geometry_object.geometry

        # Try to load mesh from path first, to take advantage of very effective
        # Panda3d model caching procedure.
        is_success = True
        mesh_path = geometry_object.meshPath
        if '\\' in mesh_path or '/' in mesh_path:
            # Assuming it is an actual path if it has a least on slash. It is
            # way faster than actually checking if the path actually exists.

            # Assimp backend used to load meshes does not support many things
            # related to paths on Windows. First, it does not support symlinks,
            # then the hard drive prefix must be `/x/` instead of `X:\`, and
            # finally backslashes must be used as delimiter instead of
            # forwardslashes.
            mesh_path = geometry_object.meshPath
            if sys.platform.startswith('win'):
                mesh_path = os.path.realpath(mesh_path)
                mesh_path = PureWindowsPath(mesh_path).as_posix()
                mesh_path = re.sub(r'^([A-Za-z]):',
                                   lambda m: "/" + m.group(1).lower(),
                                   mesh_path)
            # append a mesh
            scale = npToTuple(geometry_object.meshScale)
            self.viewer.append_mesh(*node_name, mesh_path, scale)
        elif isinstance(geom, hppfcl.ShapeBase):
            # append a primitive geometry
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
            elif isinstance(geom, (hppfcl.Convex, hppfcl.BVHModelOBBRSS)):
                # Extract vertices and faces from geometry
                if isinstance(geom, hppfcl.Convex):
                    vertices = [geom.points(i) for i in range(geom.num_points)]
                    faces = [np.array(list(geom.polygons(i)))
                             for i in range(geom.num_polygons)]
                else:
                    vertices = [geom.vertices(i)
                                for i in range(geom.num_vertices)]
                    faces = [np.array(list(geom.tri_indices(i)))
                             for i in range(geom.num_tris)]

                # Create primitive triangle geometry
                vformat = GeomVertexFormat.get_v3()
                vdata = GeomVertexData('vdata', vformat, Geom.UHStatic)
                vdata.uncleanSetNumRows(geom.num_points)
                vwriter = GeomVertexWriter(vdata, 'vertex')
                for vertex in vertices:
                    vwriter.addData3(*vertex)
                prim = GeomTriangles(Geom.UHStatic)
                for face in faces:
                    prim.addVertices(*face)
                    prim.addVertices(*face[[1, 0, 2]])  # Necessary but why ?
                obj = Geom(vdata)
                obj.addPrimitive(prim)

                # Add the primitive geometry to the scene
                geom_node = GeomNode('convex')
                geom_node.add_geom(obj)
                node = NodePath(geom_node)
                self.viewer._app.append_node(*node_name, node)
            else:
                is_success = False
        else:
            is_success = False

        # Early return if impossible to load the geometry for some reason
        if not is_success:
            warnings.warn(
                f"Unsupported geometry type for {geometry_object.name} "
                f"({type(geom)})", category=UserWarning, stacklevel=2)
            return

        # Set material color from URDF
        if color is not None:
            self.viewer.set_material(*node_name, color)
        elif geometry_object.overrideMaterial:
            color = geometry_object.meshColor
            path = geometry_object.meshTexturePath
            self.viewer.set_material(*node_name, color, path)

    def loadViewerModel(self,
                        rootNodeName: str,
                        color: Optional[np.ndarray] = None) -> None:
        """Create a group of nodes displaying the robot meshes in the viewer.
        """
        self.root_name = rootNodeName

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

    def display(self, q: np.ndarray) -> None:
        """Display the robot at configuration q in the viewer by placing all
        the bodies."""
        pin.forwardKinematics(self.model, self.data, q)

        def move(group, model, data):
            pin.updateGeometryPlacements(self.model, self.data, model, data)
            name_pose_dict = {}
            for obj in model.geometryObjects:
                oMg = data.oMg[model.getGeometryId(obj.name)]
                x, y, z, qx, qy, qz, qw = pin.SE3ToXYZQUATtuple(oMg)
                name_pose_dict[obj.name] = ((x, y, z), (qw, qx, qy, qz))
            self.viewer.move_nodes(group, name_pose_dict)

        if self.display_visuals:
            move(self.visual_group, self.visual_model, self.visual_data)

        if self.display_collisions:
            move(self.collision_group, self.collision_model,
                 self.collision_data)

    def displayCollisions(self, visibility: bool) -> None:
        """Set whether to display collision objects or not."""
        self.viewer.show_group(self.collision_group, visibility)
        self.display_collisions = visibility

    def displayVisuals(self, visibility: bool) -> None:
        """Set whether to display visual objects or not."""
        self.viewer.show_group(self.visual_group, visibility)
        self.display_visuals = visibility
