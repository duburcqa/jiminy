utilities
=========

Geometry
--------

.. autoclass:: jiminy_py.core.HeightmapType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.HeightmapFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: jiminy_py.core.sum_heightmaps

.. autofunction:: jiminy_py.core.merge_heightmaps

.. autofunction:: jiminy_py.core.discretize_heightmap

Random number generation
------------------------

.. autoclass:: jiminy_py.core.PCG32
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.PeriodicGaussianProcess
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.PeriodicFourierProcess
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.PeriodicPerlinProcess
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.RandomPerlinProcess
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: jiminy_py.core.uniform

.. autofunction:: jiminy_py.core.normal

.. autofunction:: jiminy_py.core.random_tile_ground

Pinocchio
---------

.. autoclass:: jiminy_py.core.JointModelType
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: jiminy_py.core.build_geom_from_urdf

.. autofunction:: jiminy_py.core.build_models_from_urdf

.. autofunction:: jiminy_py.core.get_joint_type

.. autofunction:: jiminy_py.core.get_joint_indices

.. autofunction:: jiminy_py.core.get_joint_position_first_index

.. autofunction:: jiminy_py.core.is_position_valid

.. autofunction:: jiminy_py.core.get_frame_indices

.. autofunction:: jiminy_py.core.get_joint_indices

.. autofunction:: jiminy_py.core.aba

.. autofunction:: jiminy_py.core.rnea

.. autofunction:: jiminy_py.core.crba

.. autofunction:: jiminy_py.core.computeKineticEnergy

.. autofunction:: jiminy_py.core.computeJMinvJt

.. autofunction:: jiminy_py.core.solveJMinvJtv

Miscellaneous
-------------

.. autoclass:: jiminy_py.core.LogicError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.OSError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.NotImplementedError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.BadControlFlow
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jiminy_py.core.LookupError
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: jiminy_py.core.array_copyto

.. autofunction:: jiminy_py.core.interpolate_positions
