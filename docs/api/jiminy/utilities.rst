utilities
=========

Timer
-----

.. doxygenclass:: jiminy::Timer
   :project: jiminy
   :members:
   :undoc-members:


IO file and Directory
---------------------

.. doxygenfunction:: jiminy::getUserDirectory
   :project: jiminy

Conversion from/to JSON
-----------------------

.. doxygenfunction:: jiminy::convertToJson(configHolder_t const &value)
   :project: jiminy

.. doxygenfunction:: jiminy::convertFromJson(Json::Value const &value)
   :project: jiminy

Geometry
--------

.. doxygenfunction:: jiminy::sumHeightmaps
   :project: jiminy

.. doxygenfunction:: jiminy::mergeHeightmaps
   :project: jiminy

.. doxygenfunction:: jiminy::discretizeHeightmap
   :project: jiminy

Random number generation
------------------------

.. doxygenfunction:: jiminy::PCG32
   :project: jiminy
   :members:

.. doxygenfunction:: jiminy::uniform
   :project: jiminy

.. doxygenfunction:: jiminy::normal
   :project: jiminy

.. doxygenfunction:: jiminy::PeriodicGaussianProcess
   :project: jiminy
   :members:

.. doxygenfunction:: jiminy::PeriodicFourierProcess
   :project: jiminy
   :members:

.. doxygenfunction:: jiminy::RandomPerlinProcess
   :project: jiminy
   :members:

.. doxygenfunction:: jiminy::PeriodicPerlinProcess
   :project: jiminy
   :members:

.. doxygenfunction:: jiminy::tiles
   :project: jiminy

Telemetry
---------

.. doxygenfunction:: jiminy::getLogFieldValue
   :project: jiminy

Pinocchio
---------

.. doxygenfunction:: jiminy::getJointNameFromPositionIndex(pinocchio::Model const &model, int32_t const &index, std::string &jointName)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointNameFromVelocityIndex(pinocchio::Model const &model, int32_t const &index, std::string &jointName)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointTypeFromIndex
   :project: jiminy

.. doxygenfunction:: jiminy::getJointTypePositionSuffixes
   :project: jiminy

.. doxygenfunction:: jiminy::getJointTypeVelocitySuffixes
   :project: jiminy

.. doxygenfunction:: jiminy::getFrameIndex
   :project: jiminy

.. doxygenfunction:: jiminy::getFrameIndices
   :project: jiminy

.. doxygenfunction:: jiminy::getJointIndex
   :project: jiminy

.. doxygenfunction:: jiminy::getJointIndices
   :project: jiminy

.. doxygenfunction:: jiminy::getJointPositionIndex(pinocchio::Model const &model, std::string const &jointName, std::vector<int32_t> &jointPositionIdx)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointsPositionIndices
   :project: jiminy

.. doxygenfunction:: jiminy::getJointVelocityIndex(pinocchio::Model const &model, std::string const &jointName, std::vector<int32_t> &jointVelocityIdx)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointVelocityIndices
   :project: jiminy

.. doxygenfunction:: jiminy::isPositionValid
   :project: jiminy

.. doxygenfunction:: jiminy::convertForceGlobalFrameToJoint
   :project: jiminy
