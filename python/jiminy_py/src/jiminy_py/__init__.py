import os as _os

# [Windows] Define the jiminy package as a site-package
# to enable global import of Eigenpy and Pinocchio
if _os.name == 'nt':
	import site as _site
	_site.addsitedir(_os.path.dirname(_os.path.realpath(__file__)))

# Import eigenpy and Pinocchio, then patch Pinocchio 
# to fix support of numpy.ndarray as eigen conversion
try:
	import eigenpy
	import pinocchio
	from . import _pinocchio_init as _patch
except ImportError:
	pass

# Import core submodule
from . import core
