import warnings
import numpy as np

import hppfcl
import pinocchio as pin

# Use numpy array by default for Eigenpy and Pinocchio incidentally
__import__('eigenpy').switchToNumpyArray()

# Disable all deprecation warnings of Pinocchio because, for now, Jiminy
# supports many releases, for which some methods have different signatures.
warnings.filterwarnings("ignore", category=pin.DeprecatedWarning)


# Fix Meshcat Viewer display method to support of numpy array return type
def display(self, q):
    pin.forwardKinematics(self.model, self.data, q)
    pin.updateGeometryPlacements(
        self.model, self.data, self.visual_model, self.visual_data)
    for visual in self.visual_model.geometryObjects:
        # Get mesh pose.
        M = self.visual_data.oMg[self.visual_model.getGeometryId(visual.name)]
        # Manage scaling
        S = np.diag(np.concatenate((visual.meshScale, np.array([1.0]))).flat)
        T = np.array(M.homogeneous).dot(S)
        # Update viewer configuration.
        self.viewer[self.getViewerNodeName(
            visual, pin.GeometryType.VISUAL)].set_transform(T)
pin.visualize.meshcat_visualizer.MeshcatVisualizer.display = display  # noqa


# Do not load the geometry of the ground is is not an actually geometry but
# rather a calculus artifact.
loadPrimitive_orig = \
    pin.visualize.gepetto_visualizer.GepettoVisualizer.loadPrimitive
def loadPrimitive(self, meshName, geometry_object):  # noqa
    geom = geometry_object.geometry
    if geometry_object.name == "ground":
        return False
    elif isinstance(geom, hppfcl.Convex):
        pts = [pin.utils.npToTuple(geom.points(geom.polygons(f)[i]))
               for f in range(geom.num_polygons) for i in range(3)]
        self.viewer.gui.addCurve(
            meshName, pts, pin.utils.npToTuple(geometry_object.meshColor))
        self.viewer.gui.setCurveMode(meshName, "TRIANGLES")
        self.viewer.gui.setLightingMode(meshName, "ON")
        self.viewer.gui.setBoolProperty(meshName, "BackfaceDrawing", True)
        return True
    else:
        return loadPrimitive_orig(self, meshName, geometry_object)
pin.visualize.gepetto_visualizer.GepettoVisualizer.loadPrimitive = \
    loadPrimitive  # noqa
