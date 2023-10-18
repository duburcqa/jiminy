#include "jiminy/core/io/serialization.h"

#define HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION
#include "hpp/fcl/serialization/BVH_model.h"  // `serialize<hpp::fcl::BVHModel>`
#undef HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION

#ifdef _MSC_VER  /* Microsoft Visual C++ -- warning level 3 */
#pragma warning(disable : 4267)  /* conversion from 'size_t' to 'unsigned int' */
#endif

// Explicit template instantiation for serialization
#define EXPL_TPL_INST_SERIALIZE_IMPL(A, ...) \
    template void serialize(A &, __VA_ARGS__ &, const unsigned int);

#define EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(...) \
    EXPL_TPL_INST_SERIALIZE_IMPL(boost::archive::binary_iarchive, __VA_ARGS__) \
    EXPL_TPL_INST_SERIALIZE_IMPL(boost::archive::binary_oarchive, __VA_ARGS__)


namespace boost::serialization
{
    template<class Archive>
    void serialize(Archive & ar,
                   pinocchio::GeometryObject & geom,
                   const unsigned int /* version */)
    {
        ar & make_nvp("name", geom.name);
        ar & make_nvp("parentFrame", geom.parentFrame);
        ar & make_nvp("parentJoint", geom.parentJoint);

        /* Manual polymorphic up-casting to avoid relying on boost
            serialization for doing it, otherwise it would conflict
            with any pinocchio bindings exposing the same objects. */
        hpp::fcl::NODE_TYPE nodeType;
        if (Archive::is_saving::value)
        {
            nodeType = geom.geometry->getNodeType();
        }
        ar & make_nvp("nodeType", nodeType);

        #define UPCAST_FROM_TYPENAME(TYPENAME, CLASS) \
        case hpp::fcl::NODE_TYPE::TYPENAME: \
            if (Archive::is_loading::value) \
            { \
                geom.geometry = std::shared_ptr<CLASS>(new CLASS); \
            } \
            ar & make_nvp("geometry", static_cast<CLASS &>(*geom.geometry)); \
            break;

        switch (nodeType)
        {
        UPCAST_FROM_TYPENAME(GEOM_TRIANGLE, hpp::fcl::TriangleP)
        UPCAST_FROM_TYPENAME(GEOM_SPHERE, hpp::fcl::Sphere)
        UPCAST_FROM_TYPENAME(GEOM_BOX, hpp::fcl::Box)
        UPCAST_FROM_TYPENAME(GEOM_CAPSULE, hpp::fcl::Capsule)
        UPCAST_FROM_TYPENAME(GEOM_CONE, hpp::fcl::Cone)
        UPCAST_FROM_TYPENAME(GEOM_CYLINDER, hpp::fcl::Cylinder)
        UPCAST_FROM_TYPENAME(GEOM_HALFSPACE, hpp::fcl::Halfspace)
        UPCAST_FROM_TYPENAME(GEOM_PLANE, hpp::fcl::Plane)
        UPCAST_FROM_TYPENAME(GEOM_ELLIPSOID, hpp::fcl::Ellipsoid)
        UPCAST_FROM_TYPENAME(GEOM_CONVEX, hpp::fcl::Convex<hpp::fcl::Triangle>)
        UPCAST_FROM_TYPENAME(BV_AABB, hpp::fcl::BVHModel<hpp::fcl::AABB>)
        UPCAST_FROM_TYPENAME(BV_OBB, hpp::fcl::BVHModel<hpp::fcl::OBB>)
        UPCAST_FROM_TYPENAME(BV_RSS, hpp::fcl::BVHModel<hpp::fcl::RSS>)
        UPCAST_FROM_TYPENAME(BV_OBBRSS, hpp::fcl::BVHModel<hpp::fcl::OBBRSS>)
        UPCAST_FROM_TYPENAME(BV_kIOS, hpp::fcl::BVHModel<hpp::fcl::kIOS>)
        UPCAST_FROM_TYPENAME(BV_KDOP16, hpp::fcl::BVHModel<hpp::fcl::KDOP<16> >)
        UPCAST_FROM_TYPENAME(BV_KDOP18, hpp::fcl::BVHModel<hpp::fcl::KDOP<18> >)
        UPCAST_FROM_TYPENAME(BV_KDOP24, hpp::fcl::BVHModel<hpp::fcl::KDOP<24> >)
        case hpp::fcl::NODE_TYPE::GEOM_OCTREE:
        case hpp::fcl::NODE_TYPE::HF_AABB:
        case hpp::fcl::NODE_TYPE::HF_OBBRSS:
        case hpp::fcl::NODE_TYPE::BV_UNKNOWN:
        case hpp::fcl::NODE_TYPE::NODE_COUNT:
        default:
            // Falling back to polymorphism
            ar & make_nvp("geometry", geom.geometry);
            break;
        }

        #undef UPCAST_FROM_TYPENAME

        ar & make_nvp("placement", geom.placement);
        ar & make_nvp("meshPath", geom.meshPath);
        ar & make_nvp("meshScale", geom.meshScale);
        ar & make_nvp("overrideMaterial", geom.overrideMaterial);
        ar & make_nvp("meshColor", geom.meshColor);
        ar & make_nvp("meshTexturePath", geom.meshTexturePath);
    }

    EXPLICIT_TEMPLATE_INSTANTIATION_SERIALIZE(pinocchio::GeometryObject)
}
