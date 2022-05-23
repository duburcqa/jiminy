
#include "jiminy/core/io/Serialization.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::TriangleP)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::Sphere)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::Box)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::Capsule)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::Cone)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::Cylinder)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::Halfspace)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::Plane)
// BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::ConvexBase)  // Convex are not serializable for now
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::AABB>)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::OBB>)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::RSS>)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::OBBRSS>)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::kIOS>)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::KDOP<16> >)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::KDOP<18> >)
BOOST_CLASS_EXPORT_IMPLEMENT(hpp::fcl::BVHModel<hpp::fcl::KDOP<24> >)

// Explicit template instantiation for serialization
#define EXPL_TPL_INST_SERIALIZE_IMPL(A, T) \
    template void serialize(A &, T &, const unsigned int);

#define EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(T) \
    EXPL_TPL_INST_SERIALIZE_IMPL(boost::archive::text_iarchive, T) \
    EXPL_TPL_INST_SERIALIZE_IMPL(boost::archive::text_oarchive, T)

namespace boost
{
    namespace serialization
    {
        template<class Archive>
        void serialize(Archive & ar,
                       pinocchio::GeometryObject & geom,
                       const unsigned int /* version */)
        {
            // Casting from BVHModelTpl to BVHModelBase is not automatically registered because the
            // serialization is made through an intermediary access `internal::BVHModelAccessor<BV>`.
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::OBBRSS>, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::AABB>, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::OBB>, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::RSS>, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::OBBRSS>, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::kIOS>, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::KDOP<16> >, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::KDOP<18> >, hpp::fcl::BVHModelBase>();
            void_cast_register<hpp::fcl::BVHModel<hpp::fcl::KDOP<24> >, hpp::fcl::BVHModelBase>();

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

        EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(pinocchio::GeometryObject)
    }
}
