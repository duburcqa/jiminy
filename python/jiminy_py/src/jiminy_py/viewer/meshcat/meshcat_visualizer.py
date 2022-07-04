import os
import math
import base64
import warnings
import xml.etree.ElementTree as Et
from typing import Optional, Any, Dict, Union, Type, Set

import numpy as np

import meshcat
from meshcat.geometry import (
    ReferenceSceneElement, Geometry, TriangularMeshGeometry)

import hppfcl
import pinocchio as pin
from pinocchio.utils import npToTuple
from pinocchio.visualize import BaseVisualizer


MsgType = Dict[str, Union[str, bytes, bool, float, 'MsgType']]


class Cone(Geometry):
    """A cone of the given height and radius. By Three.js convention, the axis
    of rotational symmetry is aligned with the y-axis.
    """
    def __init__(self, height: float, radius: float):
        super().__init__()
        self.radius = radius
        self.height = height
        self.radialSegments = 32

    def lower(self, object_data: Any) -> MsgType:
        return {
            u"uuid": self.uuid,
            u"type": u"ConeGeometry",
            u"radius": self.radius,
            u"height": self.height,
            u"radialSegments": self.radialSegments
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
        self.path = None
        self.material = None

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
                if e.tag.count('init_from')]

        # Convert textures to data URL for Three.js ColladaLoader to load them
        self.img_resources = {}
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


class MeshcatVisualizer(BaseVisualizer):
    """A Pinocchio display using Meshcat.

    Based on https://github.com/stack-of-tasks/pinocchio/blob/master/bindings/python/pinocchio/visualize/meshcat_visualizer.py
    Copyright (c) 2014-2020, CNRS
    Copyright (c) 2018-2020, INRIA
    """  # noqa: E501
    def initViewer(self,
                   viewer: meshcat.Visualizer = None,
                   loadModel: bool = False,
                   mustOpen: bool = False,
                   **kwargs: Any) -> None:
        """Start a new MeshCat server and client.
        Note: the server can also be started separately using the
        "meshcat-server" command in a terminal: this enables the server to
        remain active after the current script ends.
        """
        self.cache = set()
        self.root_name = None
        self.visual_group = None
        self.collision_group = None
        self.display_visuals = False
        self.display_collisions = False
        self.viewer = viewer

        if viewer is None:
            self.viewer = meshcat.Visualizer()

        if mustOpen:
            self.viewer.open()

        if loadModel:
            self.loadViewerModel(rootNodeName=self.model.name)

    def getViewerNodeName(self,
                          geometry_object: pin.GeometryObject,
                          geometry_type: pin.GeometryType):
        """Return the name of the geometry object inside the viewer.
        """
        if geometry_type is pin.GeometryType.VISUAL:
            return '/'.join((self.visual_group, geometry_object.name))
        elif geometry_type is pin.GeometryType.COLLISION:
            return '/'.join((self.collision_group, geometry_object.name))

    def loadPrimitive(self, geometry_object: pin.GeometryObject):
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
        elif isinstance(geom, (hppfcl.Convex, hppfcl.BVHModelBase)):
            # Extract vertices and faces from geometry
            if isinstance(geom, hppfcl.Convex):
                vertices = geom.points()
                num_faces, get_faces = geom.num_polygons, geom.polygons
            else:
                vertices = geom.vertices()
                num_faces, get_faces = geom.num_tris, geom.tri_indices
            faces = np.empty((num_faces, 3), dtype=np.int32)
            for i in range(num_faces):
                tri = get_faces(i)
                for j in range(3):
                    faces[i, j] = tri[j]

            # Create primitive triangle geometry
            obj = TriangularMeshGeometry(vertices, faces)
            geometry_object.meshScale = np.ones(3)  # It is already at scale !
        else:
            msg = "Unsupported geometry type for %s (%s)" % (
                geometry_object.name, type(geom))
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            obj = None

        return obj

    def loadMesh(self, geometry_object: pin.GeometryObject):
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

    def loadViewerGeometryObject(self,
                                 geometry_object: pin.GeometryObject,
                                 geometry_type: pin.GeometryType,
                                 color: Optional[np.ndarray] = None,
                                 Material: Type[meshcat.geometry.Material] =
                                 meshcat.geometry.MeshPhongMaterial):
        """Load a single geometry object"""
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
            msg = ("Error while loading geometry object: %s\nError message:\n"
                   "%s") % (geometry_object.name, e)
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return

        # Set material color from URDF
        material = Material()
        if color is None:
            meshColor = geometry_object.meshColor
        else:
            meshColor = color
        material.color = (int(meshColor[0] * 255) * 256 ** 2 +
                          int(meshColor[1] * 255) * 256 +
                          int(meshColor[2] * 255))

        # Add transparency, if needed.
        if float(meshColor[3]) < 1.0:
            material.transparent = True
            material.opacity = float(meshColor[3])

        # Add meshcat object to the scene
        v = self.viewer[node_name]
        if isinstance(obj, DaeMeshGeometryWithTexture):
            obj.path = v.path
            if geometry_object.overrideMaterial:
                obj.material = material
            v.window.send(obj)
        else:
            v.set_object(obj, material)

    def loadViewerModel(self,
                        rootNodeName: str,
                        color: Optional[np.ndarray] = None):
        """Load the robot in a MeshCat viewer.
        Parameters:
            rootNodeName: name to give to the robot in the viewer
            color: optional, color to give to the robot. This overwrites the
            color present in the urdf. The format is a list of four RGBA floats
            (between 0 and 1)
        """
        self.root_name = rootNodeName

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

    def display(self, q: np.ndarray):
        """Display the robot at configuration q in the viewer by placing all
        the bodies."""
        pin.forwardKinematics(self.model, self.data, q)

        if self.display_visuals:
            pin.updateGeometryPlacements(
                self.model, self.data, self.visual_model, self.visual_data)
            for i, visual in enumerate(self.visual_model.geometryObjects):
                # Get mesh pose
                M = self.visual_data.oMg[i]
                # Manage scaling
                S = np.diag(np.concatenate(
                    (visual.meshScale, np.array([1.0]))).flat)
                T = np.array(M.homogeneous).dot(S)
                # Update viewer configuration
                nodeName = self.getViewerNodeName(
                    visual, pin.GeometryType.VISUAL)
                self.viewer[nodeName].set_transform(T)

        if self.display_collisions:
            pin.updateGeometryPlacements(
                self.model, self.data, self.collision_model,
                self.collision_data)
            for i, collision in enumerate(
                    self.collision_model.geometryObjects):
                # Get mesh pose
                M = self.collision_data.oMg[i]
                # Manage scaling
                S = np.diag(np.concatenate(
                    (collision.meshScale, np.array([1.0]))).flat)
                T = np.array(M.homogeneous).dot(S)
                # Update viewer configuration
                nodeName = self.getViewerNodeName(
                    collision, pin.GeometryType.collision)
                self.viewer[nodeName].set_transform(T)

    def displayCollisions(self, visibility: bool):
        """Set whether to display collision objects or not.
        """
        self.display_collisions = visibility
        if self.collision_model is None:
            self.display_collisions = False
            return

        self.viewer[self.collision_group].set_property(
            "visible", visibility)

    def displayVisuals(self, visibility: bool):
        """Set whether to display visual objects or not.
        """
        self.display_visuals = visibility
        if self.visual_model is None:
            self.display_visuals = False
            return

        self.viewer[self.visual_group].set_property(
            "visible", visibility)
