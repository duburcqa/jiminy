
import os
import math
import warnings
import numpy as np
from typing import Optional, Any, Dict, Union, Type

import meshcat
from meshcat.geometry import Geometry, TriangularMeshGeometry, pack_numpy_array

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


class Capsule(Geometry):
    """A capsule of a given radius and height.

    Inspired from
    https://gist.github.com/aceslowman/d2fbad8b0f21656007e337543866539c,
    itself inspired from http://paulbourke.net/geometry/spherical/.
    """
    __slots__ = [
        "radius", "height", "radialSegments", "vertices", "faces", "normals"]

    def __init__(self, height: float, radius: float):
        super().__init__()
        self.radius = radius
        self.height = height
        self.radialSegments = 32
        self.build_triangles()

    def build_triangles(self) -> None:
        # Define proxy for convenience
        N = self.radialSegments

        # Initialize internal buffers
        vertices, faces = [], []

        # Top and bottom caps vertices
        for e, rng in enumerate([
                range(int(N//4) + 1), range(int(N//4), int(N//2) + 1)]):
            for i in rng:
                for j in range(N + 1):
                    theta = j * 2 * math.pi / N
                    phi = math.pi * (i / (N // 2) - 1 / 2)
                    vertex = np.empty(3)
                    vertex[0] = self.radius * math.cos(phi) * math.cos(theta)
                    vertex[1] = self.radius * math.cos(phi) * math.sin(theta)
                    vertex[2] = self.radius * math.sin(phi)
                    vertex[2] += (2.0 * (e - 0.5)) * self.height / 2
                    vertices.append(vertex)

        # Faces
        for i in range(int(N//2) + 1):
            for j in range(N):
                vec = np.array([i * (N + 1) + j,
                                i * (N + 1) + (j + 1),
                                (i + 1) * (N + 1) + (j + 1),
                                (i + 1) * (N + 1) + j])
                if (i == N//4):
                    faces.append(vec[[0, 2, 3]])
                    faces.append(vec[[0, 1, 2]])
                else:
                    faces.append(vec[[0, 1, 2]])
                    faces.append(vec[[0, 2, 3]])

        # Convert to array
        self.vertices = np.vstack(vertices).astype(np.float32)
        self.faces = np.vstack(faces).astype(np.uint32)
        self.normals = self.vertices

    def lower(self, object_data: Any) -> MsgType:
        return {
            u"uuid": self.uuid,
            u"type": u"BufferGeometry",
            u"data": {
                u"attributes": {
                    u"position": pack_numpy_array(self.vertices.T),
                    u"normal": pack_numpy_array(self.normals.T)
                },
                u"index": pack_numpy_array(self.faces.T)
            }
        }


class MeshcatVisualizer(BaseVisualizer):
    """A Pinocchio display using Meshcat.

    Based on https://github.com/stack-of-tasks/pinocchio/blob/master/bindings/python/pinocchio/visualize/meshcat_visualizer.py
    Copyright (c) 2014-2020, CNRS
    Copyright (c) 2018-2020, INRIA
    """  # noqa: E501
    def initViewer(self,
                   viewer: meshcat.Visualizer = None,
                   loadModel: bool = False,
                   mustOpen: bool = False):
        """Start a new MeshCat server and client.
        Note: the server can also be started separately using the
        "meshcat-server" command in a terminal: this enables the server to
        remain active after the current script ends.
        """
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
                          geometry_object: hppfcl.CollisionGeometry,
                          geometry_type: pin.GeometryType):
        """Return the name of the geometry object inside the viewer.
        """
        if geometry_type is pin.GeometryType.VISUAL:
            return '/'.join((self.visual_group, geometry_object.name))
        elif geometry_type is pin.GeometryType.COLLISION:
            return '/'.join((self.collision_group, geometry_object.name))

    def loadPrimitive(self, geometry_object: hppfcl.CollisionGeometry):
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
        elif isinstance(geom, (hppfcl.Convex, hppfcl.BVHModelOBBRSS)):
            # Extract vertices and faces from geometry
            if isinstance(geom, hppfcl.Convex):
                vertices = np.vstack([
                    geom.points(i) for i in range(geom.num_points)])
                faces = np.vstack([np.array(geom.polygons(i))
                                   for i in range(geom.num_polygons)])
            else:
                vertices = np.vstack([geom.vertices(i)
                                      for i in range(geom.num_vertices)])
                faces = np.vstack([np.array(geom.tri_indices(i))
                                   for i in range(geom.num_tris)])

            # Create primitive triangle geometry
            obj = TriangularMeshGeometry(vertices, faces)
            geometry_object.meshScale = np.ones(3)  # It is already at scale !
        else:
            msg = "Unsupported geometry type for %s (%s)" % (
                geometry_object.name, type(geom))
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            obj = None

        return obj

    def loadMesh(self, geometry_object: hppfcl.CollisionGeometry):
        # Mesh path is empty if Pinocchio is built without HPP-FCL bindings
        if geometry_object.meshPath == "":
            msg = ("Display of geometric primitives is supported only if "
                   "pinocchio is build with HPP-FCL bindings.")
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return None

        # Get file type from filename extension.
        _, file_extension = os.path.splitext(geometry_object.meshPath)
        if file_extension.lower() == ".dae":
            obj = meshcat.geometry.DaeMeshGeometry.from_file(
                geometry_object.meshPath)
        elif file_extension.lower() == ".obj":
            obj = meshcat.geometry.ObjMeshGeometry.from_file(
                geometry_object.meshPath)
        elif file_extension.lower() == ".stl":
            obj = meshcat.geometry.StlMeshGeometry.from_file(
                geometry_object.meshPath)
        else:
            msg = "Unknown mesh file format: {}.".format(
                geometry_object.meshPath)
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            obj = None

        return obj

    def loadViewerGeometryObject(self,
                                 geometry_object: hppfcl.CollisionGeometry,
                                 geometry_type: pin.GeometryType,
                                 color: Optional[np.ndarray] = None,
                                 Material: Type[meshcat.geometry.Material] =
                                 meshcat.geometry.MeshPhongMaterial):
        """Load a single geometry object"""
        node_name = self.getViewerNodeName(geometry_object, geometry_type)

        # Create meshcat object based on the geometry
        try:
            # Trying to load mesh preferably if available
            mesh_path = geometry_object.meshPath
            if '\\' in mesh_path or '/' in mesh_path:
                obj = self.loadMesh(geometry_object)
            elif isinstance(geometry_object.geometry, hppfcl.ShapeBase):
                obj = self.loadPrimitive(geometry_object)
            else:
                obj = None
            if obj is None:
                return
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
