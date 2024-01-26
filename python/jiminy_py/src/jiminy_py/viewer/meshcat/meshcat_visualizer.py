""" TODO: Write documentation.
"""
# pylint: disable=attribute-defined-outside-init
import os
import math
import base64
import warnings
import xml.etree.ElementTree as Et
from typing import Optional, Any, Dict, Union, Type, Set

import numpy as np

import meshcat
import meshcat.path
import meshcat.commands
from meshcat.geometry import (
    ReferenceSceneElement, Geometry, TriangularMeshGeometry)

import hppfcl
import pinocchio as pin
from pinocchio.utils import npToTuple
from pinocchio.visualize import BaseVisualizer

from ..geometry import extract_vertices_and_faces_from_geometry


MsgType = Dict[str, Union[str, bytes, bool, float, 'MsgType']]


class Cone(Geometry):
    """A cone of the given height and radius. By Three.js convention, the axis
    of rotational symmetry is aligned with the y-axis.
    """
    def __init__(self, height: float, radius: float):
        super().__init__()
        self.radius = radius
        self.height = height
        self.radial_segments = 32

    def lower(self,
              object_data: Any  # pylint: disable=unused-argument
              ) -> MsgType:
        """ TODO: Write documentation.
        """
        return {
            "uuid": self.uuid,
            "type": "ConeGeometry",
            "radius": self.radius,
            "height": self.height,
            "radialSegments": self.radial_segments
        }


class Capsule(TriangularMeshGeometry):
    """A capsule of a given radius and height.

    Inspired from
    https://gist.github.com/aceslowman/d2fbad8b0f21656007e337543866539c,
    itself inspired from http://paulbourke.net/geometry/spherical/.
    """
    def __init__(self, height: float, radius: float, num_segments: int = 32):
        # Define proxy for convenience
        N = num_segments

        # Top and bottom caps vertices
        vertices = []
        for side, rng in enumerate([
                range(int(N // 4) + 1), range(int(N // 4), int(N // 2) + 1)]):
            for i in rng:
                for j in range(N):
                    theta = j * 2 * math.pi / N
                    phi = math.pi * (i / (N // 2) - 1 / 2)
                    vertex = np.empty(3)
                    vertex[0] = radius * math.cos(phi) * math.cos(theta)
                    vertex[1] = radius * math.cos(phi) * math.sin(theta)
                    vertex[2] = radius * math.sin(phi)
                    vertex[2] += (2.0 * (side - 0.5)) * height / 2
                    vertices.append(vertex)
        vertices = np.vstack(vertices)

        # Vertex indices for faces
        faces = []
        for i in range(int(N//2) + 1):
            for j in range(N):
                vec = np.array([i * N + j,
                                i * N + (j + 1) % N,
                                (i + 1) * N + (j + 1) % N,
                                (i + 1) * N + j])
                faces.append(vec[[0, 1, 2]])
                faces.append(vec[[0, 2, 3]])
        faces = np.vstack(faces)

        # Init base class
        super().__init__(vertices, faces)


class DaeMeshGeometryWithTexture(ReferenceSceneElement):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 dae_path: str,
                 cache: Optional[Set[str]] = None) -> None:
        """Load Collada files with texture images.

        Inspired from
        https://gist.github.com/danzimmerman/a392f8eadcf1166eb5bd80e3922dbdc5
        """
        # Init base class
        super().__init__()

        # Attributes to be specified by the user
        self.path: Optional[meshcat.path.Path] = None
        self.material: Optional[meshcat.geometry.Material] = None

        # Raw file content
        dae_dir = os.path.dirname(dae_path)
        with open(dae_path, 'r') as text_file:
            self.dae_raw = text_file.read()

        # Parse the image resource in Collada file
        img_resource_paths = []
        img_lib_element = Et.parse(dae_path).find(
            "{http://www.collada.org/2005/11/COLLADASchema}library_images")
        if img_lib_element:
            img_resource_paths = [
                e.text for e in img_lib_element.iter()
                if e.tag.count('init_from') and e.text is not None]

        # Convert textures to data URL for Three.js ColladaLoader to load them
        self.img_resources: Dict[str, str] = {}
        for img_path in img_resource_paths:
            # Return empty string if already in cache
            if cache is not None:
                if img_path in cache:
                    self.img_resources[img_path] = ""
                    continue
                cache.add(img_path)

            # Encode texture in base64
            img_path_abs = img_path
            if not os.path.isabs(img_path):
                img_path_abs = os.path.normpath(
                    os.path.join(dae_dir, img_path_abs))
            if not os.path.isfile(img_path_abs):
                raise UserWarning(f"Texture '{img_path}' not found.")
            with open(img_path_abs, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read())
            img_uri = f"data:image/png;base64,{img_data.decode('utf-8')}"
            self.img_resources[img_path] = img_uri

    def lower(self) -> Dict[str, Any]:
        """Pack data into a dictionary of the format that must be passed to
        `Visualizer.window.send`.
        """
        data = {
            'type': 'set_object',
            'path': self.path.lower() if self.path is not None else "",
            'object': {
                'metadata': {'version': 4.5, 'type': 'Object'},
                'geometries': [],
                'materials': [],
                'object': {
                    'uuid': self.uuid,
                    'type': '_meshfile_object',
                    'format': 'dae',
                    'data': self.dae_raw,
                    'resources': self.img_resources
                }
            }
        }
        if self.material is not None:
            self.material.lower_in_object(data)
        return data


def update_floor(viewer: meshcat.Visualizer,
                 geom: Optional[hppfcl.CollisionGeometry] = None) -> None:
    """Display a custom collision geometry as ground profile or a flat tile
    ground floor in a given viewer instance.

    :param viewer: Meshcat viewer instance.
    :param geom: Ground profile as a collision geometry object, as provided by
                 `jiminy.discretize_heightmap`. It renders a flat tile ground
                 if not specified.
                 Optional: None by default.
    """
    # Disable the existing custom ground if any and show original flat ground
    if geom is None:
        viewer["Ground"].delete()
        viewer.window.send(meshcat.commands.SetProperty(
            "visible", True, meshcat.path.Path(("Grid",))))
        return

    # Convert provided geometry in Meshcat-specific triangle-based geometry
    vertices, faces = extract_vertices_and_faces_from_geometry(geom)
    obj = TriangularMeshGeometry(vertices, faces)
    material = meshcat.geometry.MeshLambertMaterial()
    material.color = (255 << 16) + (255 << 8) + 255

    # Disable newly created custom ground profile and hide original flat ground
    viewer["Ground"].set_object(obj, material)
    viewer.window.send(meshcat.commands.SetProperty(
        "visible", False, meshcat.path.Path(("Grid",))))


class MeshcatVisualizer(BaseVisualizer):
    """A Pinocchio display using Meshcat.

    Based on https://github.com/stack-of-tasks/pinocchio/blob/master/bindings/python/pinocchio/visualize/meshcat_visualizer.py
    Copyright (c) 2014-2020, CNRS
    Copyright (c) 2018-2020, INRIA
    """  # noqa: E501  # pylint: disable=line-too-long
    def initViewer(self,  # pylint: disable=arguments-differ
                   viewer: meshcat.Visualizer = None,
                   loadModel: bool = False,
                   mustOpen: bool = False,
                   **kwargs: Any) -> None:
        """Start a new MeshCat server and client.
        Note: the server can also be started separately using the
        "meshcat-server" command in a terminal: this enables the server to
        remain active after the current script ends.
        """
        self.cache: Set[str] = set()
        self.root_name: Optional[str] = None
        self.visual_group: Optional[str] = None
        self.collision_group: Optional[str] = None
        self.display_visuals = False
        self.display_collisions = False
        self.viewer = viewer

        if viewer is None:
            self.viewer = meshcat.Visualizer()

        if mustOpen:
            self.viewer.open()

        if loadModel:
            self.loadViewerModel(root_node_name=self.model.name)

    def getViewerNodeName(self,
                          geometry_object: pin.GeometryObject,
                          geometry_type: pin.GeometryType) -> str:
        """Return the name of the geometry object inside the viewer.
        """
        if geometry_type is pin.GeometryType.VISUAL:
            assert self.visual_group is not None
            return '/'.join((self.visual_group, geometry_object.name))
        # if geometry_type is pin.GeometryType.COLLISION:
        assert self.collision_group is not None
        return '/'.join((self.collision_group, geometry_object.name))

    def loadPrimitive(self,  # pylint: disable=invalid-name
                      geometry_object: pin.GeometryObject) -> hppfcl.ShapeBase:
        """ TODO: Write documentation.
        """
        geom = geometry_object.geometry
        if isinstance(geom, hppfcl.Capsule):
            obj = Capsule(2. * geom.halfLength, geom.radius)
        elif isinstance(geom, hppfcl.Cylinder):
            # Cylinders need to be rotated
            R = np.array([[1.,  0.,  0.,  0.],
                          [0.,  0., -1.,  0.],
                          [0.,  1.,  0.,  0.],
                          [0.,  0.,  0.,  1.]])
            RotatedCylinder = type("RotatedCylinder",
                                   (meshcat.geometry.Cylinder,),
                                   {"intrinsic_transform": lambda self: R})
            obj = RotatedCylinder(2. * geom.halfLength, geom.radius)
        elif isinstance(geom, hppfcl.Box):
            obj = meshcat.geometry.Box(npToTuple(2. * geom.halfSide))
        elif isinstance(geom, hppfcl.Sphere):
            obj = meshcat.geometry.Sphere(geom.radius)
        elif isinstance(geom, hppfcl.Cone):
            obj = Cone(2. * geom.halfLength, geom.radius)
        else:
            # Try extract raw vertices and faces from geometry
            try:
                vertices, faces = (
                    extract_vertices_and_faces_from_geometry(geom))
            except ValueError:
                warnings.warn(
                    f"Unsupported geometry type for {geometry_object.name} "
                    f"({type(geom)})",
                    category=UserWarning, stacklevel=2)
                return None

            # Create Meshcat-specific triangle-based geometry
            obj = TriangularMeshGeometry(vertices, faces)
            geometry_object.meshScale = np.ones(3)  # Already at scale !

        return obj

    def loadMesh(self,  # pylint: disable=invalid-name
                 geometry_object: pin.GeometryObject) -> ReferenceSceneElement:
        """ TODO: Write documentation.
        """
        # Mesh path is empty if Pinocchio is built without HPP-FCL bindings
        mesh_path = geometry_object.meshPath
        if mesh_path == "":
            msg = ("Display of geometric primitives is supported only if "
                   "pinocchio is build with HPP-FCL bindings.")
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return None

        # Get file type from filename extension
        _, file_extension = os.path.splitext(mesh_path)
        if file_extension.lower() == ".dae":
            obj = DaeMeshGeometryWithTexture(mesh_path, self.cache)
        elif file_extension.lower() == ".obj":
            obj = meshcat.geometry.ObjMeshGeometry.from_file(mesh_path)
        elif file_extension.lower() == ".stl":
            obj = meshcat.geometry.StlMeshGeometry.from_file(mesh_path)
        else:
            msg = f"Unknown mesh file format: {mesh_path}."
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            obj = None

        return obj

    def loadViewerGeometryObject(  # pylint: disable=invalid-name
            self,
            geometry_object: pin.GeometryObject,
            geometry_type: pin.GeometryType,
            color: Optional[np.ndarray] = None,
            material_class: Type[
                meshcat.geometry.Material] = meshcat.geometry.MeshPhongMaterial
            ) -> None:
        """Load a single geometry object"""
        # pylint: disable=broad-exception-caught
        node_name = self.getViewerNodeName(geometry_object, geometry_type)

        # Create meshcat object based on the geometry.
        try:
            # Trying to load mesh preferably if available
            mesh_path = geometry_object.meshPath
            if any(char in mesh_path for char in ('\\', '/', '.')):
                # Assuming it is an actual path if it has a least one slash.
                # It is way faster than checking the path actually exists.
                obj = self.loadMesh(geometry_object)
            else:
                obj = self.loadPrimitive(geometry_object)
        except Exception as e:
            warnings.warn(
                "Error while loading geometry object: "
                f"{geometry_object.name}\n Error message:\n{e}",
                category=UserWarning, stacklevel=2)
            return

        # Set material color from URDF
        material = material_class()
        if color is None:
            mesh_color = geometry_object.meshColor
        else:
            mesh_color = color
        material.color = ((int(mesh_color[0] * 255) << 16) +
                          (int(mesh_color[1] * 255) << 8) +
                          int(mesh_color[2] * 255))

        # Add transparency, if needed.
        if float(mesh_color[3]) < 1.0:
            material.transparent = True
            material.opacity = float(mesh_color[3])

        # Add meshcat object to the scene
        v = self.viewer[node_name]
        if isinstance(obj, DaeMeshGeometryWithTexture):
            obj.path = v.path
            if geometry_object.overrideMaterial:
                obj.material = material
            v.window.send(obj)
        else:
            v.set_object(obj, material)

    def loadViewerModel(self,  # pylint: disable=arguments-differ
                        root_node_name: str,
                        color: Optional[np.ndarray] = None) -> None:
        """Load the robot in a MeshCat viewer.

        :param root_node_name: name to give to the robot in the viewer
        :param color: Color to give to the robot. This overwrites the color
                      present in the URDF file. The format is a 1D array of
                      four RGBA floats scaled between 0.0 and 1.0.
        """
        self.root_name = root_node_name

        # Load robot visual meshes
        self.visual_group = "/".join((self.root_name, "visuals"))
        for visual in self.visual_model.geometryObjects:
            self.loadViewerGeometryObject(
                visual, pin.GeometryType.VISUAL, color)
        self.displayVisuals(True)

        # Load robot collision meshes
        self.collision_group = "/".join((self.root_name, "collisions"))
        for collision in self.collision_model.geometryObjects:
            self.loadViewerGeometryObject(
                collision, pin.GeometryType.COLLISION, color)
        self.displayCollisions(False)

        # The cache is cleared after loading every loading to avoid edge-cases
        self.cache.clear()

    def display(self,  # pylint: disable=signature-differs
                q: np.ndarray) -> None:
        """Display the robot at configuration q in the viewer by placing all
        the bodies.
        """
        pin.forwardKinematics(self.model, self.data, q)

        if self.display_visuals:
            pin.updateGeometryPlacements(
                self.model, self.data, self.visual_model, self.visual_data)
            for i, visual in enumerate(self.visual_model.geometryObjects):
                # Get mesh pose
                M = self.visual_data.oMg[i]
                # Manage scaling
                T = M.homogeneous
                T[:3, :3] *= visual.meshScale
                # Update viewer configuration
                node_name = self.getViewerNodeName(
                    visual, pin.GeometryType.VISUAL)
                self.viewer[node_name].set_transform(T)

        if self.display_collisions:
            pin.updateGeometryPlacements(
                self.model, self.data, self.collision_model,
                self.collision_data)
            for i, collision in enumerate(
                    self.collision_model.geometryObjects):
                # Get mesh pose
                M = self.collision_data.oMg[i]
                # Manage scaling
                T = M.homogeneous
                T[:3, :3] *= collision.meshScale
                # Update viewer configuration
                node_name = self.getViewerNodeName(
                    collision, pin.GeometryType.collision)
                self.viewer[node_name].set_transform(T)

    def displayCollisions(self, visibility: bool) -> None:
        """Set whether to display collision objects or not.
        """
        self.display_collisions = visibility
        if self.collision_model is None:
            self.display_collisions = False
            return

        self.viewer[self.collision_group].set_property(
            "visible", visibility)

    def displayVisuals(self, visibility: bool) -> None:
        """Set whether to display visual objects or not.
        """
        self.display_visuals = visibility
        if self.visual_model is None:
            self.display_visuals = False
            return

        self.viewer[self.visual_group].set_property(
            "visible", visibility)
