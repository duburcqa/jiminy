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


Random number generation
------------------------

.. doxygenfunction:: jiminy::resetRandomGenerators
   :project: jiminy

.. doxygenfunction:: jiminy::randUniform
   :project: jiminy

.. doxygenfunction:: jiminy::randNormal
   :project: jiminy

.. doxygenfunction:: jiminy::randVectorNormal(uint32_t const &size, float64_t const &mean, float64_t const &std)
   :project: jiminy


Telemetry
---------

.. doxygenfunction:: jiminy::getLogFieldValue
   :project: jiminy

Pinocchio
---------

.. doxygenfunction:: jiminy::getJointNameFromPositionIdx(pinocchio::Model const &model, int32_t const &idx, std::string &jointNameOut)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointNameFromVelocityIdx(pinocchio::Model const &model, int32_t const &idIn, std::string &jointNameOut)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointTypeFromIdx
   :project: jiminy

.. doxygenfunction:: jiminy::getJointTypePositionSuffixes
   :project: jiminy

.. doxygenfunction:: jiminy::getJointTypeVelocitySuffixes
   :project: jiminy

.. doxygenfunction:: jiminy::getFrameIdx
   :project: jiminy

.. doxygenfunction:: jiminy::getFramesIdx
   :project: jiminy

.. doxygenfunction:: jiminy::getJointModelIdx
   :project: jiminy

.. doxygenfunction:: jiminy::getJointsModelIdx
   :project: jiminy

.. doxygenfunction:: jiminy::getJointPositionIdx(pinocchio::Model const &model, std::string const &jointName, std::vector<int32_t> &jointPositionIdx)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointsPositionIdx
   :project: jiminy

.. doxygenfunction:: jiminy::getJointVelocityIdx(pinocchio::Model const &model, std::string const &jointName, std::vector<int32_t> &jointVelocityIdx)
   :project: jiminy

.. doxygenfunction:: jiminy::getJointsVelocityIdx
   :project: jiminy

.. doxygenfunction:: jiminy::isPositionValid
   :project: jiminy

.. doxygenfunction:: jiminy::convertForceGlobalFrameToJoint
   :project: jiminy

Math
----

.. doxygenfunction:: jiminy::clamp(Eigen::MatrixBase<DerivedType> const &data, float64_t const &minThr = -INF, float64_t const &maxThr = +INF)
   :project: jiminy
