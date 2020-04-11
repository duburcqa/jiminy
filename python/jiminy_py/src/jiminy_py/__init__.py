import os as _os
if _os.name == 'nt':
	import site as _site
	_site.addsitedir(_os.path.dirname(_os.path.realpath(__file__)))

try:
	import eigenpy as _eigenpy
	import pinocchio as _pinocchio
	from . import _pinocchio_init as _patch
except ImportError:
	pass
