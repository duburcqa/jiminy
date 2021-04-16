import pinocchio as pin


# Use numpy array by default for Eigenpy and Pinocchio incidentally
__import__('eigenpy').switchToNumpyArray()


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
