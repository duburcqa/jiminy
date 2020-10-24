import warnings

import pinocchio as pin


# Use numpy array by default for Eigenpy and Pinocchio incidentally
__import__('eigenpy').switchToNumpyArray()

# Disable all deprecation warnings of Pinocchio because, for now, Jiminy
# supports many releases, for which some methods have different signatures.
warnings.filterwarnings("ignore", category=pin.DeprecatedWarning)


# Do not load the geometry of the ground is is not an actually geometry but
# rather a calculus artifact.
loadPrimitive_orig = \
    pin.visualize.gepetto_visualizer.GepettoVisualizer.loadPrimitive
def loadPrimitive(self, meshName, geometry_object):  # noqa
    if geometry_object.name == "ground":
        return False
    else:
        return loadPrimitive_orig(self, meshName, geometry_object)
pin.visualize.gepetto_visualizer.GepettoVisualizer.loadPrimitive = \
    loadPrimitive  # noqa
