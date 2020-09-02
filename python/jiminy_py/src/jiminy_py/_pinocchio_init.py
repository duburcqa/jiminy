__import__('eigenpy').switchToNumpyArray()

# Patching pinocchio to fix support of numpy.array

import warnings
import numpy as np
from pkg_resources import parse_version as version
from math import atan2, pi, sqrt

import pinocchio as pin

warnings.filterwarnings("ignore", message="DeprecatedWarning: This function "\
    "signature is now deprecated and will be removed in future releases of Pinocchio.")

from pinocchio.rpy import npToTTuple

def npToTuple(M):
    if M.ndim == 1:
        return tuple(M.tolist())
    else:
        if M.shape[0] == 1:
            return tuple(M.tolist()[0])
        if M.shape[1] == 1:
            return tuple(M.T.tolist()[0])
        return npToTTuple(M)

pin.rpy.npToTuple = npToTuple

# `__version__` attribute exists since 2.1.1, but not properly maintained (2.4.0 and 2.4.1 are missing it...).
# On the contrary, `printVersion` has always been available and maintained.
if version(pin.printVersion()) < version("2.3.0"):
    def rotate(axis, ang):
        """
        # Transformation Matrix corresponding to a rotation about x,y or z
        eg. T = rot('x', pi / 4): rotate pi/4 rad about x axis
        """
        cood = {'x': 0, 'y': 1, 'z': 2}
        u = np.zeros((3,), dtype=np.float64)
        u[cood[axis]] = 1.0
        return np.asmatrix(pin.AngleAxis(ang, u).matrix())

    def rpyToMatrix(rpy):
        """
        # Convert from Roll, Pitch, Yaw to transformation Matrix
        """
        return rotate('z', float(rpy[2])) * rotate('y', float(rpy[1])) * rotate('x', float(rpy[0]))

    def matrixToRpy(M):
        """
        # Convert from Transformation Matrix to Roll, Pitch, Yaw
        """
        m = sqrt(M[2, 1] ** 2 + M[2, 2] ** 2)
        p = atan2(-M[2, 0], m)

        if abs(abs(p) - pi / 2) < 0.001:
            r = 0
            y = -atan2(M[0, 1], M[1, 1])
        else:
            y = atan2(M[1, 0], M[0, 0])  # alpha
            r = atan2(M[2, 1], M[2, 2])  # gamma

        return np.array([r, p, y], dtype=np.float64)

    pin.rpy.rotate = rotate
    pin.rpy.rpyToMatrix = rpyToMatrix
    pin.rpy.matrixToRpy = matrixToRpy


def display(self, q):
    """Display the robot at configuration q in the viewer by placing all the bodies."""
    pin.forwardKinematics(self.model, self.data, q)
    pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
    for visual in self.visual_model.geometryObjects:
        # Get mesh pose.
        M = self.visual_data.oMg[self.visual_model.getGeometryId(visual.name)]
        # Manage scaling
        S = np.diag(np.concatenate((visual.meshScale, np.array([1.0]))).flat)
        T = np.array(M.homogeneous).dot(S)
        # Update viewer configuration.
        self.viewer[self.getViewerNodeName(visual, pin.GeometryType.VISUAL)].set_transform(T)

pin.visualize.meshcat_visualizer.MeshcatVisualizer.display = display
