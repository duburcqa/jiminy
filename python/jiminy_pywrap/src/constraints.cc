#include "jiminy/core/robot/model.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/constraints/joint_constraint.h"
#include "jiminy/core/constraints/frame_constraint.h"
#include "jiminy/core/constraints/distance_constraint.h"
#include "jiminy/core/constraints/sphere_constraint.h"
#include "jiminy/core/constraints/wheel_constraint.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/functors.h"
#include "jiminy/python/constraints.h"


namespace jiminy
{
    // ***************************** PyConstraintVisitor ***********************************

    /* Using an intermediary class is a trick to enable defining `bp::base<...>` in conjunction
       with `bp::wrapper<...>`. */
    class AbstractConstraintImpl : public AbstractConstraintTpl<AbstractConstraintImpl>
    {
    };

    /* Explicit template specialization must appear in exactly the same namespace than its
       template declaration. */
    template<>
    const std::string AbstractConstraintTpl<AbstractConstraintImpl>::type_("UserConstraint");
}

namespace jiminy::python
{
    namespace bp = boost::python;

    class AbstractConstraintWrapper :
    public AbstractConstraintImpl,
        public bp::wrapper<AbstractConstraintImpl>
    {
    public:
        void reset(const Eigen::VectorXd & q, const Eigen::VectorXd & v)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(FunPyWrapperArgToPython(q), FunPyWrapperArgToPython(v));
            }
        }

        void computeJacobianAndDrift(const Eigen::VectorXd & q, const Eigen::VectorXd & v)
        {
            bp::override func = this->get_override("compute_jacobian_and_drift");
            if (func)
            {
                func(FunPyWrapperArgToPython(q), FunPyWrapperArgToPython(v));
            }
        }
    };

    struct PyConstraintVisitor : public bp::def_visitor<PyConstraintVisitor>
    {
    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .ADD_PROPERTY_GET_WITH_POLICY("type",
                                              &AbstractConstraintBase::getType,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_SET_WITH_POLICY("is_enabled",
                                                  &AbstractConstraintBase::getIsEnabled,
                                                  bp::return_value_policy<bp::return_by_value>(),
                                                  &PyConstraintVisitor::setIsEnable)
                .ADD_PROPERTY_GET_SET("kp",
                                      &AbstractConstraintBase::getBaumgartePositionGain,
                                      &AbstractConstraintBase::setBaumgartePositionGain)
                .ADD_PROPERTY_GET_SET("kd",
                                      &AbstractConstraintBase::getBaumgarteVelocityGain,
                                      &AbstractConstraintBase::setBaumgarteVelocityGain)
                .ADD_PROPERTY_GET_SET("baumgarte_freq",
                                      &AbstractConstraintBase::getBaumgarteFreq,
                                      &AbstractConstraintBase::setBaumgarteFreq)
                .ADD_PROPERTY_GET_WITH_POLICY("jacobian",
                                              &AbstractConstraintBase::getJacobian,
                                              bp::return_value_policy<result_converter<false>>())
                .ADD_PROPERTY_GET_WITH_POLICY("drift",
                                              &AbstractConstraintBase::getDrift,
                                              bp::return_value_policy<result_converter<false>>())
                .DEF_READONLY("lambda_c", &AbstractConstraintBase::lambda_)
                ;
            // clang-format on
        }

        static std::shared_ptr<FrameConstraint> frameConstraintFactory(
            const std::string & frameName, const bp::object & maskDoFsPy)
        {
            std::array<bool, 6> maskDoFs;
            if (maskDoFsPy.is_none())
            {
                maskDoFs = {true, true, true, true, true, true};
            }
            else
            {
                bp::extract<bp::list> maskDoFsPyExtract(maskDoFsPy);
                assert(maskDoFsPyExtract.check() && "'maskDoFsPy' must be a list.");
                bp::list maskDoFsListPy = maskDoFsPyExtract();
                assert(bp::len(maskDoFsListPy) == 6 && "'maskDoFsPy' must have length 6.");
                for (uint32_t i = 0; i < 6; ++i)
                {
                    bp::extract<bool> boolPyExtract(maskDoFsListPy[i]);
                    assert(boolPyExtract.check() && "'maskDoFsPy' elements must be bool.");
                    maskDoFs[i] = boolPyExtract();
                }
            }
            return std::make_shared<FrameConstraint>(frameName, maskDoFs);
        }

        static void setIsEnable(AbstractConstraintBase & self, bool value)
        {
            if (value)
            {
                self.enable();
            }
            else
            {
                self.disable();
            }
        }

    public:
        static void expose()
        {
            // clang-format off
            bp::class_<AbstractConstraintBase,
                       std::shared_ptr<AbstractConstraintBase>,
                       boost::noncopyable>("AbstractConstraint", bp::no_init)
                .def(PyConstraintVisitor())
                .def("reset", &AbstractConstraintBase::reset,
                              (bp::arg("self"), "q", "v"))
                .def("compute_jacobian_and_drift", &AbstractConstraintBase::computeJacobianAndDrift,
                                                   (bp::arg("self"), "q", "v"));

            bp::class_<AbstractConstraintWrapper, bp::bases<AbstractConstraintBase>,
                       std::shared_ptr<AbstractConstraintWrapper>,
                       boost::noncopyable>("BaseConstraint")
                .def_readonly("type", &AbstractConstraintWrapper::type_)
                .def("reset", bp::pure_virtual(&AbstractConstraintBase::reset))
                .def("compute_jacobian_and_drift", bp::pure_virtual(&AbstractConstraintBase::computeJacobianAndDrift));

            bp::class_<JointConstraint, bp::bases<AbstractConstraintBase>,
                       std::shared_ptr<JointConstraint>,
                       boost::noncopyable>("JointConstraint",
                       bp::init<const std::string &>(
                       (bp::arg("self"), "joint_name")))
                .def_readonly("type", &JointConstraint::type_)
                .ADD_PROPERTY_GET_WITH_POLICY("joint_name",
                                              &JointConstraint::getJointName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_index",
                                              &JointConstraint::getJointIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_SET_WITH_POLICY("reference_configuration",
                                                  &JointConstraint::getReferenceConfiguration,
                                                  bp::return_value_policy<result_converter<false>>(),
                                                  &JointConstraint::setReferenceConfiguration)
                .ADD_PROPERTY_GET_SET_WITH_POLICY("rotation_dir",
                                                  &JointConstraint::getRotationDir,
                                                  bp::return_value_policy<bp::return_by_value>(),
                                                  &JointConstraint::setRotationDir);

            bp::class_<FrameConstraint, bp::bases<AbstractConstraintBase>,
                       std::shared_ptr<FrameConstraint>,
                       boost::noncopyable>("FrameConstraint", bp::no_init)
                .def("__init__", bp::make_constructor(&PyConstraintVisitor::frameConstraintFactory,
                                 bp::default_call_policies(), (bp::arg("frame_name"),
                                                               bp::arg("mask_fixed")=bp::object())))
                .def_readonly("type", &FrameConstraint::type_)
                .ADD_PROPERTY_GET_WITH_POLICY("frame_name",
                                              &FrameConstraint::getFrameName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("frame_index",
                                              &FrameConstraint::getFrameIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("dofs_fixed",
                                              &FrameConstraint::getDofsFixed,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_SET_WITH_POLICY("reference_transform",
                                                  &FrameConstraint::getReferenceTransform,
                                                  bp::return_value_policy<result_converter<false>>(),
                                                  &FrameConstraint::setReferenceTransform)
                .ADD_PROPERTY_GET_WITH_POLICY("local_rotation",
                                              &FrameConstraint::getLocalFrame,
                                              bp::return_value_policy<result_converter<false>>())
                .def("set_normal", &FrameConstraint::setNormal);

            bp::class_<DistanceConstraint, bp::bases<AbstractConstraintBase>,
                       std::shared_ptr<DistanceConstraint>,
                       boost::noncopyable>("DistanceConstraint",
                       bp::init<const std::string &, const std::string &>(
                       (bp::arg("self"), "first_frame_name", "second_frame_name")))
                .def_readonly("type", &DistanceConstraint::type_)
                .ADD_PROPERTY_GET_WITH_POLICY("frames_names",
                                              &DistanceConstraint::getFramesNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("frame_indices",
                                              &DistanceConstraint::getFrameIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_SET_WITH_POLICY("reference_distance",
                                                  &DistanceConstraint::getReferenceDistance,
                                                  bp::return_value_policy<bp::return_by_value>(),
                                                  &DistanceConstraint::setReferenceDistance);

            bp::class_<SphereConstraint, bp::bases<AbstractConstraintBase>,
                       std::shared_ptr<SphereConstraint>,
                       boost::noncopyable>("SphereConstraint",
                       bp::init<const std::string &, double>(
                       (bp::arg("self"), "frame_name", "radius")))
                .def_readonly("type", &SphereConstraint::type_)
                .ADD_PROPERTY_GET_WITH_POLICY("frame_name",
                                              &SphereConstraint::getFrameName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("frame_index",
                                              &SphereConstraint::getFrameIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_SET_WITH_POLICY("reference_transform",
                                                  &SphereConstraint::getReferenceTransform,
                                                  bp::return_value_policy<result_converter<false>>(),
                                                  &SphereConstraint::setReferenceTransform);

            bp::class_<WheelConstraint, bp::bases<AbstractConstraintBase>,
                       std::shared_ptr<WheelConstraint>,
                       boost::noncopyable>("WheelConstraint",
                       bp::init<const std::string &, double, const Eigen::Vector3d &, const Eigen::Vector3d &>(
                       (bp::arg("self"), "frame_name", "radius", "ground_normal", "wheel_axis")))
                .def_readonly("type", &WheelConstraint::type_)
                .ADD_PROPERTY_GET_WITH_POLICY("frame_name",
                                              &WheelConstraint::getFrameName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("frame_index",
                                              &WheelConstraint::getFrameIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_SET_WITH_POLICY("reference_transform",
                                                  &WheelConstraint::getReferenceTransform,
                                                  bp::return_value_policy<result_converter<false>>(),
                                                  &WheelConstraint::setReferenceTransform);
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Constraint)

    // ***************************** PyConstraintTreeVisitor ***********************************


    struct PyConstraintTreeVisitor : public bp::def_visitor<PyConstraintTreeVisitor>
    {
    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .ADD_PROPERTY_GET("bounds_joints", &PyConstraintTreeVisitor::getBoundJoints)
                .ADD_PROPERTY_GET("contact_frames", &PyConstraintTreeVisitor::getContactFrames)
                .ADD_PROPERTY_GET("collision_bodies", &PyConstraintTreeVisitor::getCollisionBodies)
                .ADD_PROPERTY_GET("registry", &PyConstraintTreeVisitor::getRegistry)
                ;
            // clang-format on
        }

        static bp::dict getBoundJoints(ConstraintTree & self)
        {
            bp::dict constraintBoundJointsPy;
            for (auto & [name, constraint] : self.boundJoints)
            {
                constraintBoundJointsPy[name] = constraint;
            }
            return constraintBoundJointsPy;
        }

        static bp::dict getContactFrames(ConstraintTree & self)
        {
            bp::dict constraintContactFramesPy;
            for (auto & [name, constraint] : self.contactFrames)
            {
                constraintContactFramesPy[name] = constraint;
            }
            return constraintContactFramesPy;
        }

        static bp::list getCollisionBodies(ConstraintTree & self)
        {
            bp::list constraintCollisionBodies;
            for (auto & constraintCollisionBodyMap : self.collisionBodies)
            {
                bp::dict constraintCollisionBodyMapPy;
                for (auto & [name, constraint] : constraintCollisionBodyMap)
                {
                    constraintCollisionBodyMapPy[name] = constraint;
                }
                constraintCollisionBodies.append(constraintCollisionBodyMapPy);
            }
            return constraintCollisionBodies;
        }

        static bp::dict getRegistry(ConstraintTree & self)
        {
            bp::dict constraintRegistryPy;
            for (auto & [name, constraint] : self.registry)
            {
                constraintRegistryPy[name] = constraint;
            }
            return constraintRegistryPy;
        }

        static void expose()
        {
            // clang-format off
            bp::class_<ConstraintTree,
                       std::shared_ptr<ConstraintTree>,
                       boost::noncopyable>("ConstraintTree", bp::no_init)
                .def(PyConstraintTreeVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(ConstraintTree)
}
