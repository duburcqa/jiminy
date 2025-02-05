#include <cstdio>      // `std::remove`
#include <filesystem>  // `std::filesystem::temp_directory_path`
#include <regex>       // `std::regex`

#include "jiminy/core/io/serialization.h"

#include "pinocchio/serialization/archive.hpp"
#include "pinocchio/serialization/model.hpp"  // `serialize<pinocchio::Model>`
#include "pinocchio/serialization/data.hpp"   // `serialize<pinocchio::Data>`
#include "pinocchio/serialization/geometry.hpp"  // `serialize<pinocchio::GeometryObject>`, `serialize<pinocchio::CollisionPair>`

#define HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION
#include "hpp/fcl/serialization/hfield.h"     // `serialize<hpp::fcl::HeightField>`
#include "hpp/fcl/serialization/convex.h"     // `serialize<hpp::fcl::Convex<T>>`
#include "hpp/fcl/serialization/BVH_model.h"  // `serialize<hpp::fcl::BVHModel<T>>`
// #include "hpp/fcl/serialization/octree.h"  // `serialize<hpp::fcl::OcTree>`
#undef HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION

#ifdef _MSC_VER                     /* Microsoft Visual C++ -- warning level 3 */
#    pragma warning(disable : 4267) /* conversion from 'size_t' to 'unsigned int' */
#endif

#include "jiminy/core/io/file_device.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/constraints/distance_constraint.h"
#include "jiminy/core/constraints/frame_constraint.h"
#include "jiminy/core/constraints/joint_constraint.h"
#include "jiminy/core/constraints/sphere_constraint.h"
#include "jiminy/core/constraints/wheel_constraint.h"
#include "jiminy/core/robot/model.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/hardware/basic_motors.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/basic_sensors.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/pinocchio.h"

#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem/operations.hpp>  // `boost::filesystem::unique_path`

// Opt-in serialization specialization for "advanced" containers
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <boost/serialization/export.hpp>  // `BOOST_CLASS_EXPORT`

// These includes are required for template instantiation of a custom binary archive.
// See official documentation fr details about `boost::serialization` library:
// https://www.boost.org/doc/libs/1_84_0/libs/serialization/doc/serialization.html
// https://www.boost.org/doc/libs/1_84_0/libs/serialization/doc/traits.html
#include <boost/archive/impl/archive_serializer_map.ipp>
#include <boost/archive/impl/basic_binary_oprimitive.ipp>
#include <boost/archive/impl/basic_binary_oarchive.ipp>
#include <boost/archive/impl/basic_binary_iprimitive.ipp>
#include <boost/archive/impl/basic_binary_iarchive.ipp>


using namespace jiminy;


// Explicit template instantiation for serialization
#define EXPL_TPL_INST_SERIALIZE_IMPL(A, ...)                      \
    template void JIMINY_TEMPLATE_INSTANTIATION_DLLAPI serialize( \
        A &, __VA_ARGS__ &, const unsigned int);

#define EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(...)                  \
    EXPL_TPL_INST_SERIALIZE_IMPL(stateful_binary_oarchive, __VA_ARGS__) \
    EXPL_TPL_INST_SERIALIZE_IMPL(stateful_binary_iarchive, __VA_ARGS__)

#define EXPL_TPL_INST_LOAD_CONSTRUCT_IMPL(A, ...)                           \
    template void JIMINY_TEMPLATE_INSTANTIATION_DLLAPI load_construct_data( \
        A &, __VA_ARGS__ *, const unsigned int);

#define EXPL_TPL_INST_SAVE_CONSTRUCT_IMPL(A, ...)                           \
    template void JIMINY_TEMPLATE_INSTANTIATION_DLLAPI save_construct_data( \
        A &, const __VA_ARGS__ *, const unsigned int);

#define EXPLICIT_TEMPLATE_INSTANTIATION_LOAD_CONSTRUCT(...) \
    EXPL_TPL_INST_LOAD_CONSTRUCT_IMPL(stateful_binary_iarchive, __VA_ARGS__)

#define EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(...)             \
    EXPL_TPL_INST_SAVE_CONSTRUCT_IMPL(stateful_binary_oarchive, __VA_ARGS__) \
    EXPL_TPL_INST_LOAD_CONSTRUCT_IMPL(stateful_binary_iarchive, __VA_ARGS__)

// **************************** jiminy::stateful_binary_(o,i)archive *************************** //

namespace boost::archive
{
    template class detail::archive_serializer_map<stateful_binary_oarchive>;
    template class basic_binary_oprimitive<stateful_binary_oarchive,
                                           std::ostream::char_type,
                                           std::ostream::traits_type>;
    template class basic_binary_oarchive<stateful_binary_oarchive>;
    template class binary_oarchive_impl<stateful_binary_oarchive,
                                        std::ostream::char_type,
                                        std::ostream::traits_type>;

    template class detail::archive_serializer_map<stateful_binary_iarchive>;
    template class basic_binary_iprimitive<stateful_binary_iarchive,
                                           std::ostream::char_type,
                                           std::ostream::traits_type>;
    template class basic_binary_iarchive<stateful_binary_iarchive>;
    template class binary_iarchive_impl<stateful_binary_iarchive,
                                        std::ostream::char_type,
                                        std::ostream::traits_type>;
}

// *************************** saveToBinary, loadFromBinary internals ************************** //

namespace jiminy
{
    template<typename T, typename... Args>
    std::string saveToBinaryImpl(const T & obj, Args &&... args)
    {
        std::ostringstream os;
        {
            stateful_binary_oarchive oa(os);
            oa.state_ = std::forward_as_tuple(std::forward<Args>(args)...);
            oa << obj;
            return os.str();
        }
    }

    template<typename T, typename... Args>
    void loadFromBinaryImpl(T & obj, const std::string & str, Args &&... args)
    {
        std::istringstream is(str);
        {
            stateful_binary_iarchive ia(is);
            ia.state_ = std::forward_as_tuple(std::forward<Args>(args)...);
            ia >> obj;
        }
    }
}

// ********************************* pinocchio::GeometryObject ********************************* //

namespace boost::serialization
{
    template<class Archive>
    void load_construct_data(
        Archive & /* ar */, pinocchio::GeometryObject * geomPtr, const unsigned int /* version */)
    {
        ::new (geomPtr) pinocchio::GeometryObject("", 0, 0, {nullptr}, pinocchio::SE3::Identity());
    }

    template<class Archive>
    void
    serialize(Archive & ar, pinocchio::GeometryObject & geom, const unsigned int /* version */)
    {
        /* Casting from BVHModelTpl / ConvexTpl to BVHModelBase / ConvexBase is not automatically
           registered because the serialization is made through intermediary accessors
           `internal::BVHModelAccessor<BV>` / `internal::ConvexAccessor<PolygonT>`. */
        void_cast_register<hpp::fcl::Convex<hpp::fcl::Triangle>, hpp::fcl::ConvexBase>();
        void_cast_register<hpp::fcl::Convex<hpp::fcl::Quadrilateral>, hpp::fcl::ConvexBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::AABB>, hpp::fcl::BVHModelBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::OBB>, hpp::fcl::BVHModelBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::RSS>, hpp::fcl::BVHModelBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::OBBRSS>, hpp::fcl::BVHModelBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::kIOS>, hpp::fcl::BVHModelBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::KDOP<16>>, hpp::fcl::BVHModelBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::KDOP<18>>, hpp::fcl::BVHModelBase>();
        void_cast_register<hpp::fcl::BVHModel<hpp::fcl::KDOP<24>>, hpp::fcl::BVHModelBase>();

        ar & make_nvp("name", geom.name);
        ar & make_nvp("parentFrame", geom.parentFrame);
        ar & make_nvp("parentJoint", geom.parentJoint);
        ar & make_nvp("geometry", geom.geometry);
        ar & make_nvp("placement", geom.placement);
        ar & make_nvp("meshPath", geom.meshPath);
        ar & make_nvp("meshScale", geom.meshScale);
        ar & make_nvp("overrideMaterial", geom.overrideMaterial);
        ar & make_nvp("meshColor", geom.meshColor);
        ar & make_nvp("meshTexturePath", geom.meshTexturePath);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::CollisionGeometry)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::ShapeBase)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::TriangleP)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Sphere)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Box)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Capsule)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Cone)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Cylinder)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Halfspace)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Ellipsoid)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Plane)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::ConvexBase)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Convex<hpp::fcl::Triangle>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::Convex<hpp::fcl::Quadrilateral>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::HeightField<hpp::fcl::AABB>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::HeightField<hpp::fcl::OBB>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::HeightField<hpp::fcl::OBBRSS>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModelBase)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::AABB>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::OBB>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::RSS>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::OBBRSS>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::kIOS>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::KDOP<16>>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::KDOP<18>>)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::BVHModel<hpp::fcl::KDOP<24>>)
    // EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(hpp::fcl::OcTree)

    EXPLICIT_TEMPLATE_INSTANTIATION_LOAD_CONSTRUCT(pinocchio::GeometryObject)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(pinocchio::GeometryObject)
}

BOOST_CLASS_EXPORT(hpp::fcl::CollisionGeometry)
BOOST_CLASS_EXPORT(hpp::fcl::ShapeBase)
BOOST_CLASS_EXPORT(hpp::fcl::TriangleP)
BOOST_CLASS_EXPORT(hpp::fcl::Sphere)
BOOST_CLASS_EXPORT(hpp::fcl::Box)
BOOST_CLASS_EXPORT(hpp::fcl::Capsule)
BOOST_CLASS_EXPORT(hpp::fcl::Cone)
BOOST_CLASS_EXPORT(hpp::fcl::Cylinder)
BOOST_CLASS_EXPORT(hpp::fcl::Halfspace)
BOOST_CLASS_EXPORT(hpp::fcl::Ellipsoid)
BOOST_CLASS_EXPORT(hpp::fcl::Plane)
BOOST_CLASS_EXPORT(hpp::fcl::Convex<hpp::fcl::Triangle>)
BOOST_CLASS_EXPORT(hpp::fcl::Convex<hpp::fcl::Quadrilateral>)
BOOST_CLASS_EXPORT(hpp::fcl::HeightField<hpp::fcl::AABB>)
BOOST_CLASS_EXPORT(hpp::fcl::HeightField<hpp::fcl::OBB>)
BOOST_CLASS_EXPORT(hpp::fcl::HeightField<hpp::fcl::OBBRSS>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::AABB>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::OBB>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::RSS>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::OBBRSS>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::kIOS>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::KDOP<16>>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::KDOP<18>>)
BOOST_CLASS_EXPORT(hpp::fcl::BVHModel<hpp::fcl::KDOP<24>>)
// BOOST_CLASS_EXPORT(hpp::fcl::OcTree)

BOOST_CLASS_EXPORT(pinocchio::GeometryObject)

// ********************************** pinocchio::GeometryModel ********************************* //

namespace boost::serialization
{
    template<class Archive>
    void
    serialize(Archive & ar, pinocchio::GeometryModel & model, const unsigned int /* version */)
    {
        ar & make_nvp("ngeoms", model.ngeoms);
        ar & make_nvp("geometryObjects", model.geometryObjects);
        ar & make_nvp("collisionPairs", model.collisionPairs);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(pinocchio::GeometryModel)
}

BOOST_CLASS_EXPORT(pinocchio::GeometryModel)

// ********************************* jiminy::HeightmapFunction ********************************* //

namespace boost::serialization
{
    template<class Archive>
    void
    save(Archive & /* ar */, const HeightmapFunction & /* fun */, const unsigned int /* version */)
    {
    }

    template<class Archive>
    void load(Archive & /* ar */, HeightmapFunction & fun, const unsigned int /* version */)
    {
        fun = [](const Eigen::Vector2d & /* xy */,
                 double & height,
                 std::optional<Eigen::Ref<Eigen::Vector3d>> normal)
        {
            height = 0.0;
            if (normal.has_value())
            {
                normal.value() = Eigen::Vector3d::UnitZ();
            }
        };
    }

    template<class Archive>
    void serialize(Archive & ar, HeightmapFunction & fun, const unsigned int version)
    {
        split_free(ar, fun, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(HeightmapFunction)
}

BOOST_CLASS_EXPORT(HeightmapFunction)

// ******************************* jiminy::FlexibilityJointConfig ****************************** //

namespace boost::serialization
{
    template<class Archive>
    void serialize(Archive & ar, FlexibilityJointConfig & config, const unsigned int /* version */)
    {
        ar & make_nvp("frameName", config.frameName);
        ar & make_nvp("stiffness", config.stiffness);
        ar & make_nvp("damping", config.damping);
        ar & make_nvp("inertia", config.inertia);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(FlexibilityJointConfig)
}

BOOST_CLASS_EXPORT(FlexibilityJointConfig)

// ******************************* jiminy::AbstractConstraintBase ****************************** //

namespace boost::serialization
{
    template<class Archive>
    void
    save(Archive & ar, const AbstractConstraintBase & constraint, const unsigned int /* version */)
    {
        bool isEnabled = constraint.getIsEnabled();
        ar << make_nvp("is_enabled", isEnabled);
        double baumgartePositionGain = constraint.getBaumgartePositionGain();
        ar << make_nvp("baumgarte_position_gain", baumgartePositionGain);
        double baumgarteVelocityGain = constraint.getBaumgarteVelocityGain();
        ar << make_nvp("baumgarte_velocity_gain", baumgarteVelocityGain);
    }

    template<class Archive>
    void load(Archive & ar, AbstractConstraintBase & constraint, const unsigned int /* version */)
    {
        bool isEnabled;
        ar >> make_nvp("is_enabled", isEnabled);
        if (isEnabled)
        {
            constraint.enable();
        }
        else
        {
            constraint.disable();
        }
        double baumgartePositionGain;
        ar >> make_nvp("baumgarte_position_gain", baumgartePositionGain);
        constraint.setBaumgartePositionGain(baumgartePositionGain);
        double baumgarteVelocityGain;
        ar >> make_nvp("baumgarte_velocity_gain", baumgarteVelocityGain);
        constraint.setBaumgarteVelocityGain(baumgarteVelocityGain);
    }

    template<class Archive>
    void serialize(Archive & ar, AbstractConstraintBase & constraint, const unsigned int version)
    {
        split_free(ar, constraint, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(AbstractConstraintBase)
}

BOOST_CLASS_EXPORT(AbstractConstraintBase)

// ********************************* jiminy::DistanceConstraint ******************************** //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const DistanceConstraint * constraintPtr, const unsigned int /* version */)
    {
        ar << make_nvp("frame_names", constraintPtr->getFrameNames());
    }

    template<class Archive>
    void
    save(Archive & ar, const DistanceConstraint & constraint, const unsigned int /* version */)
    {
        ar << make_nvp("abstract_constraint_base",
                       base_object<const AbstractConstraintBase>(constraint));

        double distanceRef = constraint.getReferenceDistance();
        ar << make_nvp("reference_distance", distanceRef);
    }

    template<class Archive>
    void load_construct_data(
        Archive & ar, DistanceConstraint * constraintPtr, const unsigned int /* version */)
    {
        std::array<std::string, 2> frameNames;
        ar >> make_nvp("frame_names", frameNames);
        ::new (constraintPtr) DistanceConstraint(frameNames[0], frameNames[1]);
    }

    template<class Archive>
    void load(Archive & ar, DistanceConstraint & constraint, const unsigned int /* version */)
    {
        ar >>
            make_nvp("abstract_constraint_base", base_object<AbstractConstraintBase>(constraint));

        double distanceRef;
        ar >> make_nvp("reference_distance", distanceRef);
        constraint.setReferenceDistance(distanceRef);
    }

    template<class Archive>
    void serialize(Archive & ar, DistanceConstraint & constraint, const unsigned int version)
    {
        split_free(ar, constraint, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(DistanceConstraint)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(DistanceConstraint)
}

BOOST_CLASS_EXPORT(DistanceConstraint)

// ********************************** jiminy::FrameConstraint ********************************** //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const FrameConstraint * constraintPtr, const unsigned int /* version */)
    {
        ar << make_nvp("frame_name", constraintPtr->getFrameName());
        ar << make_nvp("dofs_fixed", constraintPtr->getDofsFixed());
    }

    template<class Archive>
    void save(Archive & ar, const FrameConstraint & constraint, const unsigned int /* version */)
    {
        ar << make_nvp("abstract_constraint_base",
                       base_object<const AbstractConstraintBase>(constraint));

        ar << make_nvp("reference_transform", constraint.getReferenceTransform());
        ar << make_nvp("local_frame", constraint.getLocalFrame());
    }

    template<class Archive>
    void load_construct_data(
        Archive & ar, FrameConstraint * constraintPtr, const unsigned int /* version */)
    {
        std::string frameName;
        std::vector<uint32_t> dofsFixed;
        ar >> make_nvp("frame_name", frameName);
        ar >> make_nvp("dofs_fixed", dofsFixed);
        std::array<bool, 6> maskDoFs{true, true, true, true, true, true};
        for (uint32_t i : dofsFixed)
        {
            maskDoFs[i] = true;
        }
        ::new (constraintPtr) FrameConstraint(frameName, maskDoFs);
    }

    template<class Archive>
    void load(Archive & ar, FrameConstraint & constraint, const unsigned int /* version */)
    {
        ar >>
            make_nvp("abstract_constraint_base", base_object<AbstractConstraintBase>(constraint));

        pinocchio::SE3 transformRef;
        ar >> make_nvp("reference_transform", transformRef);
        constraint.setReferenceTransform(transformRef);

        Eigen::Matrix3d rotationLocal;
        ar >> make_nvp("local_frame", rotationLocal);
        constraint.setNormal(rotationLocal.col(2));
    }

    template<class Archive>
    void serialize(Archive & ar, FrameConstraint & constraint, const unsigned int version)
    {
        split_free(ar, constraint, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(FrameConstraint)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(FrameConstraint)
}

BOOST_CLASS_EXPORT(FrameConstraint)

// ********************************** jiminy::JointConstraint ********************************** //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const JointConstraint * constraintPtr, const unsigned int /* version */)
    {
        ar << make_nvp("joint_name", constraintPtr->getJointName());
    }

    template<class Archive>
    void save(Archive & ar, const JointConstraint & constraint, const unsigned int /* version */)
    {
        ar << make_nvp("abstract_constraint_base",
                       base_object<const AbstractConstraintBase>(constraint));

        ar << make_nvp("reference_configuration", constraint.getReferenceConfiguration());
        bool isReversed = constraint.getRotationDir();
        ar << make_nvp("rotation_dir", isReversed);
    }

    template<class Archive>
    void load_construct_data(
        Archive & ar, JointConstraint * constraintPtr, const unsigned int /* version */)
    {
        std::string jointName;
        ar >> make_nvp("joint_name", jointName);
        ::new (constraintPtr) JointConstraint(jointName);
    }

    template<class Archive>
    void load(Archive & ar, JointConstraint & constraint, const unsigned int /* version */)
    {
        ar >>
            make_nvp("abstract_constraint_base", base_object<AbstractConstraintBase>(constraint));

        Eigen::VectorXd referenceConfiguration;
        ar >> make_nvp("reference_configuration", referenceConfiguration);
        constraint.setReferenceConfiguration(referenceConfiguration);

        bool isReversed;
        ar >> make_nvp("rotation_dir", isReversed);
        constraint.setRotationDir(isReversed);
    }

    template<class Archive>
    void serialize(Archive & ar, JointConstraint & constraint, const unsigned int version)
    {
        split_free(ar, constraint, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(JointConstraint)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(JointConstraint)
}

BOOST_CLASS_EXPORT(JointConstraint)

// ********************************** jiminy::SphereConstraint ********************************* //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const SphereConstraint * constraintPtr, const unsigned int /* version */)
    {
        ar << make_nvp("frame_name", constraintPtr->getFrameName());
        double sphereRadius = constraintPtr->getRadius();
        ar << make_nvp("radius", sphereRadius);
        ar << make_nvp("normal", constraintPtr->getNormal());
    }

    template<class Archive>
    void save(Archive & ar, const SphereConstraint & constraint, const unsigned int /* version */)
    {
        ar << make_nvp("abstract_constraint_base",
                       base_object<const AbstractConstraintBase>(constraint));

        ar << make_nvp("reference_transform", constraint.getReferenceTransform());
    }

    template<class Archive>
    void load_construct_data(
        Archive & ar, SphereConstraint * constraintPtr, const unsigned int /* version */)
    {
        std::string frameName;
        ar >> make_nvp("frame_name", frameName);
        double sphereRadius;
        ar >> make_nvp("radius", sphereRadius);
        Eigen::Vector3d groundNormal;
        ar >> make_nvp("normal", groundNormal);
        ::new (constraintPtr) SphereConstraint(frameName, sphereRadius, groundNormal);
    }

    template<class Archive>
    void load(Archive & ar, SphereConstraint & constraint, const unsigned int /* version */)
    {
        ar >>
            make_nvp("abstract_constraint_base", base_object<AbstractConstraintBase>(constraint));

        pinocchio::SE3 transformRef;
        ar >> make_nvp("reference_transform", transformRef);
        constraint.setReferenceTransform(transformRef);
    }

    template<class Archive>
    void serialize(Archive & ar, SphereConstraint & constraint, const unsigned int version)
    {
        split_free(ar, constraint, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(SphereConstraint)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(SphereConstraint)
}

BOOST_CLASS_EXPORT(SphereConstraint)

// ********************************** jiminy::WheelConstraint ********************************** //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const WheelConstraint * constraintPtr, const unsigned int /* version */)
    {
        ar << make_nvp("frame_name", constraintPtr->getFrameName());
        double wheelRadius = constraintPtr->getRadius();
        ar << make_nvp("radius", wheelRadius);
        ar << make_nvp("normal", constraintPtr->getNormal());
        ar << make_nvp("wheel_axis", constraintPtr->getWheelAxis());
    }

    template<class Archive>
    void save(Archive & ar, const WheelConstraint & constraint, const unsigned int /* version */)
    {
        ar << make_nvp("abstract_constraint_base",
                       base_object<const AbstractConstraintBase>(constraint));

        ar << make_nvp("reference_transform", constraint.getReferenceTransform());
    }

    template<class Archive>
    void load_construct_data(
        Archive & ar, WheelConstraint * constraintPtr, const unsigned int /* version */)
    {
        std::string frameName;
        ar >> make_nvp("frame_name", frameName);
        double wheelRadius;
        ar >> make_nvp("radius", wheelRadius);
        Eigen::Vector3d groundNormal;
        ar >> make_nvp("normal", groundNormal);
        Eigen::Vector3d wheelAxis;
        ar >> make_nvp("normal", wheelAxis);
        ::new (constraintPtr) WheelConstraint(frameName, wheelRadius, groundNormal, wheelAxis);
    }

    template<class Archive>
    void load(Archive & ar, WheelConstraint & constraint, const unsigned int /* version */)
    {
        ar >>
            make_nvp("abstract_constraint_base", base_object<AbstractConstraintBase>(constraint));

        pinocchio::SE3 transformRef;
        ar >> make_nvp("reference_transform", transformRef);
        constraint.setReferenceTransform(transformRef);
    }

    template<class Archive>
    void serialize(Archive & ar, WheelConstraint & constraint, const unsigned int version)
    {
        split_free(ar, constraint, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(WheelConstraint)
    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(WheelConstraint)
}

BOOST_CLASS_EXPORT(WheelConstraint)

// *************************************** jiminy::Model *************************************** //

namespace boost::serialization
{
    template<class Archive>
    void save(Archive & ar, const Model & model, const unsigned int /* version */)
    {
        // Load arguments from archive state if available, otherwise use conservative default
        bool isPersistent = true;
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            if (ar.state_.has_value())
            {
                try
                {
                    std::tie(isPersistent) = std::any_cast<std::tuple<bool &>>(ar.state_);
                }
                catch (const std::bad_any_cast & e)
                {
                    JIMINY_WARNING("Failed to parse user-specified Model serialization arguments. "
                                   "Using default values.");
                }
            }
        }

        // Early return if not initalized
        bool isInitialized = model.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Backup URDF data
        ar << make_nvp("urdf_data", model.getUrdfAsString());

        // Backup 'has_freeflyer' option
        bool hasFreeflyer = model.getHasFreeflyer();
        ar << make_nvp("has_freeflyer", hasFreeflyer);

        // Backup mesh package lookup directories
        ar << make_nvp("mesh_package_dirs", model.getMeshPackageDirs());

        /* Copy the theoretical model then remove extra collision frames.
           Note that `removeCollisionBodies` is not called to avoid altering the robot. */
        std::vector<std::string> collisionConstraintNames;
        const ConstraintTree & constraints = model.getConstraints();
        pinocchio::Model pinocchioModelTh = model.pinocchioModelTh_;
        for (const std::string & name : model.getCollisionBodyNames())
        {
            for (const pinocchio::GeometryObject & geom : model.collisionModelTh_.geometryObjects)
            {
                if (model.pinocchioModel_.frames[geom.parentFrame].name == name &&
                    constraints.exist(geom.name, ConstraintRegistryType::COLLISION_BODIES))
                {
                    const pinocchio::FrameIndex frameIndex =
                        getFrameIndex(pinocchioModelTh, geom.name);
                    pinocchioModelTh.frames.erase(
                        std::next(pinocchioModelTh.frames.begin(), frameIndex));
                    pinocchioModelTh.nframes--;
                }
            }
        }

        // Backup theoretical and extended simulation models
        ar << make_nvp("pinocchio_model_th", pinocchioModelTh);
        ar << make_nvp("pinocchio_model", model.pinocchioModel_);

        /* Backup the Pinocchio GeometryModel for collisions and visuals if requested.
           Note that it may fail because of missing serialization methods for complex geometry
           types (although unlikely at the time being), or just because it cannot fit into RAM
           memory. */
        auto serializeGeometrySafe =
            [&ar, isUrdfDataEmpty = model.getUrdfAsString().empty(), isPersistent](
                const char * name, const pinocchio::GeometryModel & geometry)
        {
            // Try serializing geometry model if persistence is requested
            bool isValidSave = false;
            try
            {
                if (isPersistent)
                {
                    ar << make_nvp(name, geometry);
                    isValidSave = true;
                }
            }
            catch (const std::exception & e)
            {
                std::string msg = toString("Failed to serialize geometry model '", name, "'.");
                if (isUrdfDataEmpty)
                {
                    msg += "\nIt will be impossible to replay log files because no URDF file is "
                           "available as fallback.";
                }
                msg += "\nRaised from exception: ";
                JIMINY_WARNING(msg, e.what());
            }

            // Fallback to adding dummy geometry if the true one was not save
            if (!isValidSave)
            {
                pinocchio::GeometryModel dummyGeometry{};
                ar << make_nvp(name, dummyGeometry);
            }

            // Keep track of whether or not the true geometry was saved
            ar << make_nvp(toString("is_", name, "_valid").c_str(), isValidSave);
        };
        serializeGeometrySafe("collision_model_th", model.collisionModelTh_);
        serializeGeometrySafe("visual_model_th", model.visualModelTh_);

        // Backup user-specified collision body and frame names
        ar << make_nvp("collision_body_names", model.getCollisionBodyNames());
        ar << make_nvp("contact_frame_names", model.getContactFrameNames());

        /* Backup user-specified constraints.
           Although serializing `std::vector`s is supported natively, it is preferable to loop over
           all constraints manually in order to serialize the one for which it works, and just
           print warning and move on when it is not. */
        const std::shared_ptr<AbstractConstraintBase> dummyConstraintPtr{};
        std::size_t numUserConstraints = constraints.user.size();
        ar << make_nvp("num_user_constraints", numUserConstraints);
        for (std::size_t i = 0; i < numUserConstraints; ++i)
        {
            /* Note that it is not possible to serialize `std::pair` directly because serialization
            failure of the motor would corrupt the whole archive state with half-backed data. */
            const std::string name = toString("constraint_", i);
            const auto & [constraintName, constraintPtr] = constraints.user[i];
            ar << make_nvp(toString(name, "_name").c_str(), constraintName);
            try
            {
                ar << make_nvp(toString(name, "_ptr").c_str(), constraintPtr);
            }
            catch (const boost::archive::archive_exception & e)
            {
                ar << make_nvp(toString(name, "_ptr").c_str(), dummyConstraintPtr);
                JIMINY_WARNING("Failed to serialize constraint '",
                               constraints.user[i].first,
                               "'. It will be missing when loading the robot from log."
                               "\nRaised from exception: ",
                               e.what());
            }
        }

        // Backup options
        ar << make_nvp("model_options", model.getOptions());
    }

    template<class Archive>
    void load_construct_data(Archive & ar, Model * modelPtr, const unsigned int /* version */)
    {
        // Create instance
        ::new (modelPtr) Model();

        /* Tell the archive to start managing a shared_ptr.
           Note that this must be done manually here because, when a shared pointer is
           de-serialized, a raw pointer is first created. The latter only becomes managed by a
           shared pointer at the very end of the process, after `load` has been called. This means
           that any complex logic that requires having access to the shared pointer managing the
           object will break. This scenario is very common for classes that derive from
           `std::enable_shared_from_this` since the underlying shared pointer managing them is
           directly accessible through `shared_from_this`. In particular, `jiminy::Model` relies on
           this mechanism when calling `initialize` to attach constraints.
           One option to get around this issue is to specialize `serialize` for `std::shared_ptr`
           directly. This approach is not considered hacky since it is how `boost::serialization`
           has been designed to work in the first place. Yet, this approach breaks polymorphism as
           base type stored in a shared pointer is actually not a base of a derived type stored in
           a shared pointer.
           Another option is to ask the archive to create a shared pointer managing the raw pointer
           earlier than it would normally occurs. This approach is robust because it is part of how
           `boost:serialization` tracking mechanism works internally, so the lifetime should get
           managed properly. This is a approach that has been preferred here. See:
           https://github.com/boostorg/serialization/blob/develop/include/boost/serialization/shared_ptr.hpp
        */
        std::shared_ptr<Model> modelSharedPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(modelSharedPtr, modelPtr);
    }

    template<class Archive>
    void load(Archive & ar, Model & model, const unsigned int /* version */)
    {
        // Load arguments from archive state if available, otherwise use conservative default
        std::optional<std::string> meshPathDir = std::nullopt;
        std::vector<std::string> meshPackageDirs = {};
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            if (ar.state_.has_value())
            {
                try
                {
                    std::tie(meshPathDir, meshPackageDirs) =
                        std::any_cast<std::tuple<const std::optional<std::string> &,
                                                 const std::vector<std::string> &>>(ar.state_);
                }
                catch (const std::bad_any_cast & e)
                {
                    JIMINY_WARNING("Failed to parse user-specified Model de-serialization "
                                   "arguments. Using default values.");
                }
            }
        }

        // Make sure that model is managed by a shared ptr
        try
        {
            (void)model.shared_from_this();
        }
        catch (std::bad_weak_ptr & e)
        {
            JIMINY_THROW(bad_control_flow, "Model must be managed by a std::shared_ptr.");
        }

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Load URDF data
        std::string urdfData;
        ar >> make_nvp("urdf_data", urdfData);

        // Overwrite the common root of all absolute mesh paths in URDF by `meshPathDir`
        if (meshPathDir.has_value())
        {
            // Extract all mesh paths that are absolute path
            constexpr std::string_view meshTag{"<mesh filename="};
            constexpr std::string_view packagePrefix{"package://"};
            const std::regex meshPathRegex(toString(meshTag, "\"(.*)\""));
            auto meshPathBegin =
                std::sregex_iterator(urdfData.begin(), urdfData.end(), meshPathRegex);
            auto meshPathEnd = std::sregex_iterator();
            std::vector<std::filesystem::path> meshPaths;
            for (std::sregex_iterator meshPathIt = meshPathBegin; meshPathIt != meshPathEnd;
                 ++meshPathIt)
            {
                const std::string meshPath = meshPathIt->str();
                if (!meshPath.compare(0, packagePrefix.size(), packagePrefix))
                {
                    meshPaths.emplace_back(meshPath);
                }
            }

            if (!meshPaths.empty())
            {
                // Get the common root of all mesh paths
                std::string meshPathDirOrig = meshPaths[0].parent_path().string();
                for (const auto & meshPath : meshPaths)
                {
                    const std::string meshPathParent = meshPath.parent_path().string();
                    const auto meshPathDirMatchEnd = std::mismatch(meshPathDirOrig.begin(),
                                                                   meshPathDirOrig.end(),
                                                                   meshPathParent.begin())
                                                         .first;
                    meshPathDirOrig = std::string(meshPathDirOrig.begin(), meshPathDirMatchEnd);
                }

                // Override the common root of all absolute mesh paths
                const std::regex meshPathDirRegex(toString(meshTag, "\"", meshPathDirOrig));
                urdfData = std::regex_replace(
                    urdfData, meshPathDirRegex, toString(meshTag, "\"", meshPathDir.value()));
            }
        }

        // Load 'has_freeflyer' flag
        bool hasFreeflyer;
        ar >> make_nvp("has_freeflyer", hasFreeflyer);

        // Load mesh package lookup directories and append user-specified fallback directories
        std::vector<std::string> meshPackageDirs_;
        ar >> make_nvp("mesh_package_dirs", meshPackageDirs_);
        meshPackageDirs_.insert(
            meshPackageDirs_.end(), meshPackageDirs.begin(), meshPackageDirs.end());

        // Load theoretical and extended simulation models
        pinocchio::Model pinocchioModelTh, pinocchioModel;
        ar >> make_nvp("pinocchio_model_th", pinocchioModelTh);
        ar >> make_nvp("pinocchio_model", pinocchioModel);

        // Load the visual and collision theoretical models
        bool isValidCollisionModelTh, isValidVisualModelTh;
        pinocchio::GeometryModel collisionModelTh, visualModelTh;
        ar >> make_nvp("collision_model_th", collisionModelTh);
        ar >> make_nvp("is_collision_model_th_valid", isValidCollisionModelTh);
        ar >> make_nvp("visual_model_th", visualModelTh);
        ar >> make_nvp("is_visual_model_th_valid", isValidVisualModelTh);

        // Initialize model from persistent data if possible, otherwise fallback to URDF loading
        if ((isValidCollisionModelTh && isValidVisualModelTh) || urdfData.empty())
        {
            // Display warning if collision and/or visual model are not available
            if (!(isValidCollisionModelTh && isValidVisualModelTh))
            {
                JIMINY_WARNING("The log is not persistent and the model was not initialized via a "
                               "URDF file. Impossible to load collision and visual models. ");
            }

            // Initialize model from theoretical model
            model.initialize(pinocchioModelTh, collisionModelTh, visualModelTh);

            /* Hack model to overwrite URDF data if available.
               This is fragile and may break in the future. The only robust way to fix this is
               modifying the declaration of `jiminy::Model` by adding templated friend declaration
               with this current method for it to get direct access to its private members. This is
               not worth the effort in this case, since this field is optional in the first place.
               See: https://stackoverflow.com/a/30595430/4820605 */
            const_cast<std::string &>(model.getUrdfAsString()) = urdfData;
        }
        else
        {
            // Write urdf data in temporary file
            const std::string urdfPath = (std::filesystem::temp_directory_path() /
                                          boost::filesystem::unique_path("%%%%%%.urdf").native())
                                             .generic_string();
            if (!std::ofstream(urdfPath).put('a'))
            {
                JIMINY_THROW(std::ios_base::failure,
                             "Impossible to create temporary URDF file. Make sure that you have "
                             "write permissions on system temporary directory: ",
                             urdfPath);
            }
            FileDevice urdfFile(urdfPath);
            urdfFile.open(OpenMode::WRITE_ONLY);
            urdfFile.resize(static_cast<int64_t>(urdfData.size()));
            urdfFile.write(urdfData);
            urdfFile.close();

            // Initialize model from URDF
            model.initialize(urdfPath, hasFreeflyer, meshPackageDirs_);

            // Delete temporary file
            std::remove(urdfPath.c_str());

            /* Restore the theoretical model.
               Note that it also restores manually-added frames since they are not tracked
               internally for now. Note that it is also necessary to restore the extended
               simulation model otherwise adding collision bodies and contact points would fail. */
            model.pinocchioModel_ = pinocchioModelTh;
            model.pinocchioModelTh_ = pinocchioModelTh;
            model.pinocchioDataTh_ = pinocchio::Data(pinocchioModelTh);
        }

        // Restore user-specified collision body and frame names
        std::vector<std::string> collisionBodyNames, contactFrameNames;
        ar >> make_nvp("collision_body_names", collisionBodyNames);
        ar >> make_nvp("contact_frame_names", contactFrameNames);
        model.addCollisionBodies(collisionBodyNames, false);
        model.addContactPoints(contactFrameNames);

        // Restore user-specified constraints
        std::size_t numUserConstraints;
        ar >> make_nvp("num_user_constraints", numUserConstraints);
        for (std::size_t i = 0; i < numUserConstraints; ++i)
        {
            std::string constraintName;
            std::shared_ptr<AbstractConstraintBase> constraintPtr;
            const std::string name = toString("constraint_", i);
            ar >> make_nvp(toString(name, "_name").c_str(), constraintName);
            ar >> make_nvp(toString(name, "_ptr").c_str(), constraintPtr);
            if (constraintPtr)
            {
                model.addConstraint(constraintName, constraintPtr);
            }
        }

        // Load options
        GenericConfig options;
        ar >> make_nvp("model_options", options);
        model.setOptions(options);

        // Restore extended simulation model
        model.pinocchioModel_ = pinocchioModel;
        model.pinocchioData_ = pinocchio::Data(pinocchioModel);
    }

    template<class Archive>
    void serialize(Archive & ar, Model & model, const unsigned int version)
    {
        split_free(ar, model, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(Model)
    EXPLICIT_TEMPLATE_INSTANTIATION_LOAD_CONSTRUCT(Model)
}

BOOST_CLASS_EXPORT(Model)

// ********************************* jiminy::AbstractMotorBase ********************************* //

namespace boost::serialization
{
    template<class Archive>
    void serialize(
        Archive & /* ar */, AbstractMotorBase & /* motor */, const unsigned int /* version */)
    {
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(AbstractMotorBase)
}

BOOST_CLASS_EXPORT(AbstractMotorBase)

// ************************************ jiminy::SimpleMotor ************************************ //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const SimpleMotor * motorPtr, const unsigned int /* version */)
    {
        // Save constructor arguments
        ar << make_nvp("name", motorPtr->getName());
    }

    template<class Archive>
    void save(Archive & ar, const SimpleMotor & motor, const unsigned int /* version */)
    {
        // Save base
        ar << make_nvp("abstract_motor_base", base_object<AbstractMotorBase>(motor));

        // Early return if not initalized
        bool isInitialized = motor.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Save initialization data
        ar << make_nvp("joint_name", motor.getJointName());
    }

    template<class Archive>
    void
    load_construct_data(Archive & ar, SimpleMotor * motorPtr, const unsigned int /* version */)
    {
        // Load constructor arguments
        std::string name;
        ar >> make_nvp("name", name);

        // Create instance
        ::new (motorPtr) SimpleMotor(name);
    }

    template<class Archive>
    void load(Archive & ar, SimpleMotor & motor, const unsigned int /* version */)
    {
        // Tell the archive to start managing a shared_ptr.
        // Note that it is tricky to expose `load` to the user because it would require having
        // access to the shared pointer managing the motor, along with the robot to which the motor
        // should be attached. The only way to specify them is through the archive state.
        // To get around this issue, we make sure that motors can only be serialized/de-serialized
        // through the robot to which they are attached. This way, memory is guaranteed to be
        // heap-allocated using `new` operator, and therefore requesting the archive to manage a
        // shared_ptr of the motor instance is safe. */
        std::shared_ptr<SimpleMotor> motorPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(motorPtr, &motor);

        // Load base
        ar >> make_nvp("abstract_motor_base", base_object<AbstractMotorBase>(motor));

        // Attach motor
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            std::shared_ptr<Robot> robot = std::any_cast<std::shared_ptr<Robot>>(ar.state_);
            robot->attachMotor(motorPtr);
        }

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Load initialization data but initialize motor only if attached
        std::string jointName;
        ar >> make_nvp("joint_name", jointName);
        if (motor.getIsAttached())
        {
            motor.initialize(jointName);
        }
    }

    template<class Archive>
    void serialize(Archive & ar, SimpleMotor & motor, const unsigned int version)
    {
        /* It is not possible to finish initialization of a motor before attaching it to a robot.

           * One option would be to pass the robot to which the sensor must be attached via the
             archive state. However, it requires forcing serialization of motors through shared
             pointers rather than references. This is undesirable because in principle motors do
             not require being storing in shared pointer, unlike `jiminy::Model`.
           * Another option is to return a post-loading callback that must be called after
             attaching the motor to finish initialization. This options is more generic but
             defer the responsibility to finish initialization at `jiminy::Robot` deserialization
             level, which is not ideal either. Moreover, it requires being able to downcast
             `jiminy::AbstractMotorBase` pointers into their actual most derived type, which is
             fairly complex to implement.
           * Yet another option is skipping `load` entirely if the motor is not attached, so that
             it can be called manually after attaching the motor. This approach is very similar to
             the previous one, except that it works with any kind of archive and not just jiminy-
             centric `stateful_binary_(i,o)archive`, which is mainly intended at providing
             optimizations and fallbacks rather than being essential for serialization.

           In the end, the less painful of all these options seems to be the first one. */
        split_free(ar, motor, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(SimpleMotor)
    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(SimpleMotor)
}

BOOST_CLASS_EXPORT(SimpleMotor)

// ********************************* jiminy::AbstractSensorBase ******************************** //

namespace boost::serialization
{
    template<class Archive>
    void serialize(
        Archive & /* ar */, AbstractSensorBase & /* motor */, const unsigned int /* version */)
    {
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(AbstractSensorBase)
}

BOOST_CLASS_EXPORT(AbstractSensorBase)

// ************************************* jiminy::ImuSensor ************************************* //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const ImuSensor * sensorPtr, const unsigned int /* version */)
    {
        // Save constructor arguments
        ar << make_nvp("name", sensorPtr->getName());
    }

    template<class Archive>
    void save(Archive & ar, const ImuSensor & sensor, const unsigned int /* version */)
    {
        // Save base
        ar << make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Early return if not initalized
        bool isInitialized = sensor.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        /* Save initialization data.
           TODO: maybe `sensorRotationBiasInv_` should also be restored. */
        ar << make_nvp("frame_name", sensor.getFrameName());
    }

    template<class Archive>
    void load_construct_data(Archive & ar, ImuSensor * sensorPtr, const unsigned int /* version */)
    {
        // Load constructor arguments
        std::string name;
        ar >> make_nvp("name", name);

        // Create instance
        ::new (sensorPtr) ImuSensor(name);
    }

    template<class Archive>
    void load(Archive & ar, ImuSensor & sensor, const unsigned int /* version */)
    {
        // Tell the archive to start managing a shared_ptr
        std::shared_ptr<ImuSensor> sensorPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(sensorPtr, &sensor);

        // Load base
        ar >> make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Attach sensor
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            std::shared_ptr<Robot> robot = std::any_cast<std::shared_ptr<Robot>>(ar.state_);
            robot->attachSensor(sensorPtr);
        }

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Load initialization data but initialize sensor only if attached
        std::string frameName;
        ar >> make_nvp("frame_name", frameName);
        if (sensor.getIsAttached())
        {
            sensor.initialize(frameName);
        }
    }

    template<class Archive>
    void serialize(Archive & ar, ImuSensor & sensor, const unsigned int version)
    {
        split_free(ar, sensor, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(ImuSensor)
    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(ImuSensor)
}

BOOST_CLASS_EXPORT(ImuSensor)

// *********************************** jiminy::ContactSensor *********************************** //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const ContactSensor * sensorPtr, const unsigned int /* version */)
    {
        // Save constructor arguments
        ar << make_nvp("name", sensorPtr->getName());
    }

    template<class Archive>
    void save(Archive & ar, const ContactSensor & sensor, const unsigned int /* version */)
    {
        // Save base
        ar << make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Early return if not initalized
        bool isInitialized = sensor.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Save initialization data
        ar << make_nvp("frame_name", sensor.getFrameName());
    }

    template<class Archive>
    void
    load_construct_data(Archive & ar, ContactSensor * sensorPtr, const unsigned int /* version */)
    {
        // Load constructor arguments
        std::string name;
        ar >> make_nvp("name", name);

        // Create instance
        ::new (sensorPtr) ContactSensor(name);
    }

    template<class Archive>
    void load(Archive & ar, ContactSensor & sensor, const unsigned int /* version */)
    {
        // Tell the archive to start managing a shared_ptr
        std::shared_ptr<ContactSensor> sensorPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(sensorPtr, &sensor);

        // Load base
        ar >> make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Attach sensor
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            std::shared_ptr<Robot> robot = std::any_cast<std::shared_ptr<Robot>>(ar.state_);
            robot->attachSensor(sensorPtr);
        }

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Load initialization data but initialize sensor only if attached
        std::string frameName;
        ar >> make_nvp("frame_name", frameName);
        if (sensor.getIsAttached())
        {
            sensor.initialize(frameName);
        }
    }

    template<class Archive>
    void serialize(Archive & ar, ContactSensor & sensor, const unsigned int version)
    {
        split_free(ar, sensor, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(ContactSensor)
    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(ContactSensor)
}

BOOST_CLASS_EXPORT(ContactSensor)

// ************************************ jiminy::ForceSensor ************************************ //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const ForceSensor * sensorPtr, const unsigned int /* version */)
    {
        // Save constructor arguments
        ar << make_nvp("name", sensorPtr->getName());
    }

    template<class Archive>
    void save(Archive & ar, const ForceSensor & sensor, const unsigned int /* version */)
    {
        // Save base
        ar << make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Early return if not initalized
        bool isInitialized = sensor.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Save initialization data
        ar << make_nvp("frame_name", sensor.getFrameName());
    }

    template<class Archive>
    void
    load_construct_data(Archive & ar, ForceSensor * sensorPtr, const unsigned int /* version */)
    {
        // Load constructor arguments
        std::string name;
        ar >> make_nvp("name", name);

        // Create instance
        ::new (sensorPtr) ForceSensor(name);
    }

    template<class Archive>
    void load(Archive & ar, ForceSensor & sensor, const unsigned int /* version */)
    {
        // Tell the archive to start managing a shared_ptr
        std::shared_ptr<ForceSensor> sensorPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(sensorPtr, &sensor);

        // Load base
        ar >> make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Attach sensor
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            std::shared_ptr<Robot> robot = std::any_cast<std::shared_ptr<Robot>>(ar.state_);
            robot->attachSensor(sensorPtr);
        }

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Load initialization data but initialize sensor only if attached
        std::string frameName;
        ar >> make_nvp("frame_name", frameName);
        if (sensor.getIsAttached())
        {
            sensor.initialize(frameName);
        }
    }

    template<class Archive>
    void serialize(Archive & ar, ForceSensor & sensor, const unsigned int version)
    {
        split_free(ar, sensor, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(ForceSensor)
    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(ForceSensor)
}

BOOST_CLASS_EXPORT(ForceSensor)

// ******************************** jiminy::EncoderSensor ********************************* //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const EncoderSensor * sensorPtr, const unsigned int /* version */)
    {
        // Save constructor arguments
        ar << make_nvp("name", sensorPtr->getName());
    }

    template<class Archive>
    void save(Archive & ar, const EncoderSensor & sensor, const unsigned int /* version */)
    {
        // Save base
        ar << make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Early return if not initalized
        bool isInitialized = sensor.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Save initialization data
        const bool isJointSide = sensor.getMotorName().empty();
        ar << make_nvp("is_joint_side", isJointSide);
        if (isJointSide)
        {
            ar << make_nvp("joint_name", sensor.getJointName());
        }
        else
        {
            ar << make_nvp("motor_name", sensor.getMotorName());
        }
    }

    template<class Archive>
    void
    load_construct_data(Archive & ar, EncoderSensor * sensorPtr, const unsigned int /* version */)
    {
        // Load constructor arguments
        std::string name;
        ar >> make_nvp("name", name);

        // Create instance
        ::new (sensorPtr) EncoderSensor(name);
    }

    template<class Archive>
    void load(Archive & ar, EncoderSensor & sensor, const unsigned int /* version */)
    {
        // Tell the archive to start managing a shared_ptr
        std::shared_ptr<EncoderSensor> sensorPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(sensorPtr, &sensor);

        // Load base
        ar >> make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Attach sensor
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            std::shared_ptr<Robot> robot = std::any_cast<std::shared_ptr<Robot>>(ar.state_);
            robot->attachSensor(sensorPtr);
        }

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Load initialization data but initialize sensor only if attached
        bool isJointSide;
        ar >> make_nvp("is_joint_side", isJointSide);
        std::string motorOrJointName;
        if (isJointSide)
        {
            ar >> make_nvp("joint_name", motorOrJointName);
        }
        else
        {
            ar >> make_nvp("motor_name", motorOrJointName);
        }
        if (sensor.getIsAttached())
        {
            sensor.initialize(motorOrJointName, isJointSide);
        }
    }

    template<class Archive>
    void serialize(Archive & ar, EncoderSensor & sensor, const unsigned int version)
    {
        split_free(ar, sensor, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(EncoderSensor)
    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(EncoderSensor)
}

BOOST_CLASS_EXPORT(EncoderSensor)

// ************************************ jiminy::EffortSensor *********************************** //

namespace boost::serialization
{
    template<class Archive>
    void save_construct_data(
        Archive & ar, const EffortSensor * sensorPtr, const unsigned int /* version */)
    {
        // Save constructor arguments
        ar << make_nvp("name", sensorPtr->getName());
    }

    template<class Archive>
    void save(Archive & ar, const EffortSensor & sensor, const unsigned int /* version */)
    {
        // Save base
        ar << make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Early return if not initalized
        bool isInitialized = sensor.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Save initialization data
        ar << make_nvp("motor_name", sensor.getMotorName());
    }

    template<class Archive>
    void
    load_construct_data(Archive & ar, EffortSensor * sensorPtr, const unsigned int /* version */)
    {
        // Load constructor arguments
        std::string name;
        ar >> make_nvp("name", name);

        // Create instance
        ::new (sensorPtr) EffortSensor(name);
    }

    template<class Archive>
    void load(Archive & ar, EffortSensor & sensor, const unsigned int /* version */)
    {
        // Tell the archive to start managing a shared_ptr
        std::shared_ptr<EffortSensor> sensorPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(sensorPtr, &sensor);

        // Load base
        ar >> make_nvp("abstract_sensor_base", base_object<AbstractSensorBase>(sensor));

        // Attach sensor
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            std::shared_ptr<Robot> robot = std::any_cast<std::shared_ptr<Robot>>(ar.state_);
            robot->attachSensor(sensorPtr);
        }

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Load initialization data but initialize sensor only if attached
        std::string motorName;
        ar >> make_nvp("motor_name", motorName);
        if (sensor.getIsAttached())
        {
            sensor.initialize(motorName);
        }
    }

    template<class Archive>
    void serialize(Archive & ar, EffortSensor & sensor, const unsigned int version)
    {
        split_free(ar, sensor, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(EffortSensor)
    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(EffortSensor)
}

BOOST_CLASS_EXPORT(EffortSensor)

// *************************************** jiminy::Robot *************************************** //

namespace boost::serialization
{
    template<class Archive>
    void
    save_construct_data(Archive & ar, const Robot * robotPtr, const unsigned int /* version */)
    {
        // Save constructor arguments
        ar << make_nvp("name", robotPtr->getName());
    }

    template<class Archive>
    void save(Archive & ar, const Robot & robot, const unsigned int /* version */)
    {
        // Save base model
        ar << make_nvp("model", base_object<Model>(robot));

        // Early return if not initalized
        bool isInitialized = robot.getIsInitialized();
        ar << make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Backup all the motors using failsafe approach
        const std::shared_ptr<AbstractMotorBase> dummyMotorPtr{};
        const Robot::WeakMotorVector & motors = robot.getMotors();
        std::size_t nmotors = robot.nmotors();
        ar << make_nvp("nmotors", nmotors);
        for (std::size_t i = 0; i < nmotors; ++i)
        {
            auto motor = std::const_pointer_cast<AbstractMotorBase>(motors[i].lock());
            const std::string name = toString("motor_", i);
            try
            {
                ar << make_nvp(name.c_str(), motor);
            }
            catch (const boost::archive::archive_exception & e)
            {
                ar << make_nvp(name.c_str(), dummyMotorPtr);
                JIMINY_WARNING("Failed to serialize motor '",
                               motor->getName(),
                               "'. It will be missing when loading the robot from log."
                               "\nRaised from exception: ",
                               e.what());
            }
        }

        // Backup all the sensors using failsafe approach
        const std::shared_ptr<AbstractSensorBase> dummySensorPtr{};
        const Robot::WeakSensorTree & sensors = robot.getSensors();
        const std::size_t nSensorTypes = sensors.size();
        ar << make_nvp("nsensortypes", nSensorTypes);
        auto sensorsGroupIt = sensors.cbegin();
        for (std::size_t i = 0; i < nSensorTypes; ++sensorsGroupIt, ++i)
        {
            bool hasFailed = false;
            const auto & [sensorsGroupName, sensorsGroup] = *sensorsGroupIt;
            const std::size_t nSensors = sensorsGroup.size();
            ar << make_nvp(toString("type_", i, "_nsensors").c_str(), nSensors);
            for (std::size_t j = 0; j < nSensors; ++j)
            {
                auto sensor = std::const_pointer_cast<AbstractSensorBase>(sensorsGroup[j].lock());
                const std::string name = toString("type_", i, "_sensor_", j);
                try
                {
                    if (!hasFailed)
                    {
                        ar << make_nvp(name.c_str(), sensor);
                    }
                }
                catch (const boost::archive::archive_exception & e)
                {
                    if (!hasFailed)
                    {
                        JIMINY_WARNING("Failed to serialize sensors of type '",
                                       sensorsGroupName,
                                       "'. They will be missing when loading the robot from log."
                                       "\nRaised from exception: ",
                                       e.what());
                    }
                    hasFailed = true;
                }
                if (hasFailed)
                {
                    ar << make_nvp(name.c_str(), dummySensorPtr);
                }
            }
        }

        // Backup options
        ar << make_nvp("robot_options", robot.getOptions());
    }

    template<class Archive>
    void load_construct_data(Archive & ar, Robot * robotPtr, const unsigned int /* version */)
    {
        // Load constructor arguments
        std::string name;
        ar >> make_nvp("name", name);

        // Create instance
        ::new (robotPtr) Robot(name);

        // Tell the archive to start managing a shared_ptr
        std::shared_ptr<Robot> robotSharedPtr;
        shared_ptr_helper<std::shared_ptr> & h =
            ar.template get_helper<shared_ptr_helper<std::shared_ptr>>(shared_ptr_helper_id);
        h.reset(robotSharedPtr, robotPtr);
    }

    template<class Archive>
    void load(Archive & ar, Robot & robot, const unsigned int /* version */)
    {
        // Load base model
        ar >> make_nvp("model", base_object<Model>(robot));

        // Early return if not initalized
        bool isInitialized;
        ar >> make_nvp("is_initialized", isInitialized);
        if (!isInitialized)
        {
            return;
        }

        // Backup the already restored extended simulation model
        const pinocchio::Model pinocchioModel = robot.pinocchioModel_;

        /* Overwrite archive state with the shared pointer managing the robot.
           Doing this will allow for attaching the motors and sensors when loading, then
           therefore finish initializing them completely. */
        if constexpr (std::is_base_of_v<jiminy::archive::AnyState, Archive>)
        {
            ar.state_ = robot.shared_from_this();
        }

        // Restore attached motors
        std::size_t nmotors;
        ar >> make_nvp("nmotors", nmotors);
        for (std::size_t i = 0; i < nmotors; ++i)
        {
            std::shared_ptr<AbstractMotorBase> motorPtr;
            ar >> make_nvp(toString("motor_", i).c_str(), motorPtr);
        }

        // Restore attached sensors
        std::size_t nSensorTypes;
        ar >> make_nvp("nsensortypes", nSensorTypes);
        for (std::size_t i = 0; i < nSensorTypes; ++i)
        {
            std::size_t nSensors;
            ar >> make_nvp(toString("type_", i, "_nsensors").c_str(), nSensors);
            for (std::size_t j = 0; j < nSensors; ++j)
            {
                std::shared_ptr<AbstractSensorBase> sensorPtr;
                ar >> make_nvp(toString("type_", i, "_sensor_", j).c_str(), sensorPtr);
            }
        }

        /* Load robot options.
           Note that options associated with motors and sensors that were impossible to load must
           be removed, otherwise setting options will fail. */
        GenericConfig robotOptions;
        ar >> make_nvp("robot_options", robotOptions);
        GenericConfig & motorsOptions = boost::get<GenericConfig>(robotOptions.at("motors"));
        // FIXME: Replace by `std::erase_if` when moving to C++20
        const Robot::MotorVector & motors = robot.getMotors();
        auto motorsOptionsIt = motorsOptions.begin();
        while (motorsOptionsIt != motorsOptions.end())
        {
            if (std::find_if(motors.begin(),
                             motors.end(),
                             [&motorName = motorsOptionsIt->first](const auto & motor)
                             { return motor->getName() == motorName; }) == motors.end())
            {
                motorsOptionsIt = motorsOptions.erase(motorsOptionsIt);
            }
            else
            {
                ++motorsOptionsIt;
            }
        }
        GenericConfig & sensorsOptions = boost::get<GenericConfig>(robotOptions.at("sensors"));
        // FIXME: Replace by `std::erase_if` when moving to C++20
        const Robot::SensorTree & sensors = robot.getSensors();
        auto sensorsOptionsIt = sensorsOptions.begin();
        while (sensorsOptionsIt != sensorsOptions.end())
        {
            if (sensors.find(sensorsOptionsIt->first) == sensors.end())
            {
                sensorsOptionsIt = sensorsOptions.erase(sensorsOptionsIt);
            }
            else
            {
                ++sensorsOptionsIt;
            }
        }
        robot.setOptions(robotOptions);

        // Restore the extended simulation model once again
        robot.pinocchioModel_ = pinocchioModel;
    }

    template<class Archive>
    void serialize(Archive & ar, Robot & robot, const unsigned int version)
    {
        split_free(ar, robot, version);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(Robot)
    EXPLICIT_TEMPLATE_INSTANTIATION_SAVE_LOAD_CONSTRUCT(Robot)
}

BOOST_CLASS_EXPORT(Robot)

namespace jiminy
{
    std::string saveToBinary(const std::shared_ptr<const jiminy::Robot> & robot, bool isPersistent)
    {
        return saveToBinaryImpl(robot, isPersistent);
    }

    void loadFromBinary(std::shared_ptr<jiminy::Robot> & robot,
                        const std::string & data,
                        const std::optional<std::string> & meshPathDir,
                        const std::vector<std::string> & meshPackageDirs)
    {
        loadFromBinaryImpl(robot, data, meshPathDir, meshPackageDirs);
    }
}
