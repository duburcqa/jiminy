#ifndef JIMINY_SERIALIZATION_TPP
#define JIMINY_SERIALIZATION_TPP

#include <sstream>

#include "pinocchio/multibody/fcl.hpp"          // `pinocchio::CollisionPair`
#include "pinocchio/multibody/geometry.hpp"     // `pinocchio::GeometryModel`
#include "pinocchio/serialization/model.hpp"    // `serialize<pinocchio::Model>`

#define HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION
#include "hpp/fcl/serialization/collision_object.h"  // `serialize<hpp::fcl::CollisionGeometry>`
#include "hpp/fcl/serialization/geometric_shapes.h"  // `serialize<hpp::fcl::ShapeBase>`
#include "hpp/fcl/serialization/BVH_model.h"         // `serialize<hpp::fcl::BVHModel>`

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


namespace jiminy
{
    template<typename T>
    std::string saveToBinary(T const & obj)
    {
        std::ostringstream os;
        {
            boost::archive::binary_oarchive oa(os, boost::archive::no_header);
            oa << obj;
            return os.str();
        }
    }

    template<typename T>
    void loadFromBinary(T & obj, std::string const & str)
    {
        std::istringstream is(str);
        {
            boost::archive::binary_iarchive ia(is, boost::archive::no_header);
            ia >> obj;
        }
    }
}

#ifdef _MSC_VER
namespace Eigen { namespace internal {
template<> struct traits<boost::archive::xml_iarchive> {enum {Flags=0};};
template<> struct traits<boost::archive::text_iarchive> {enum {Flags=0};};
template<> struct traits<boost::archive::binary_iarchive> {enum {Flags=0};};
} }
#endif


namespace boost
{
  namespace serialization
  {
        // ================ Copied from `hpp/fcl/serialization/eigen.h` ====================
        #if !(PINOCCHIO_VERSION_AT_LEAST(2,6,0))

        template <class Archive, typename PlainObjectBase, int MapOptions, typename StrideType>
        void save(Archive & ar, const Eigen::Map<PlainObjectBase,MapOptions,StrideType> & m, const unsigned int /*version*/)
        {
            Eigen::DenseIndex rows(m.rows()), cols(m.cols());
            if (PlainObjectBase::RowsAtCompileTime == Eigen::Dynamic)
            {
                ar << BOOST_SERIALIZATION_NVP(rows);
            }
            if (PlainObjectBase::ColsAtCompileTime == Eigen::Dynamic)
            {
                ar << BOOST_SERIALIZATION_NVP(cols);
            }
            ar << make_nvp("data", make_array(m.data(), static_cast<size_t>(m.size())));
        }

        template <class Archive, typename PlainObjectBase, int MapOptions, typename StrideType>
        void load(Archive & ar, Eigen::Map<PlainObjectBase,MapOptions,StrideType> & m, const unsigned int /*version*/)
        {
            Eigen::DenseIndex rows = PlainObjectBase::RowsAtCompileTime, cols = PlainObjectBase::ColsAtCompileTime;
            if (PlainObjectBase::RowsAtCompileTime == Eigen::Dynamic)
            {
                ar >> BOOST_SERIALIZATION_NVP(rows);
            }
            if (PlainObjectBase::ColsAtCompileTime == Eigen::Dynamic)
            {
                ar >> BOOST_SERIALIZATION_NVP(cols);
            }
            m.resize(rows, cols);
            ar >> make_nvp("data", make_array(m.data(), static_cast<size_t>(m.size())));
        }

        template <class Archive, typename PlainObjectBase, int MapOptions, typename StrideType>
        void serialize(Archive & ar, Eigen::Map<PlainObjectBase,MapOptions,StrideType> & m, const unsigned int version)
        {
            split_free(ar,m,version);
        }

        #endif
        // =================================================================================

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Sphere * spherePtr,
                                 const unsigned int /* version */)
        {
            ::new(spherePtr) hpp::fcl::Sphere(0.0);
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::TriangleP * trianglePPtr,
                                 const unsigned int /* version */)
        {
            ::new(trianglePPtr) hpp::fcl::TriangleP(
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Capsule * capsulePtr,
                                 const unsigned int /* version */)
        {
            ::new(capsulePtr) hpp::fcl::Capsule(0.0, 0.0);
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Cone * conePtr,
                                 const unsigned int /* version */)
        {
            ::new(conePtr) hpp::fcl::Cone(0.0, 0.0);
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Cylinder * cylinderPtr,
                                 const unsigned int /* version */)
        {
            ::new(cylinderPtr) hpp::fcl::Cylinder(0.0, 0.0);
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 pinocchio::GeometryObject * geomPtr,
                                 const unsigned int /* version */)
        {
            ::new(geomPtr) pinocchio::GeometryObject(
                "", 0, 0, {nullptr}, pinocchio::SE3::Identity());
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 pinocchio::CollisionPair * collisionPairPtr,
                                 const unsigned int /* version */)
        {
            ::new(collisionPairPtr) pinocchio::CollisionPair(0, 1);
        }

        template<class Archive>
        void serialize(Archive & ar,
                       pinocchio::GeometryObject & geom,
                       const unsigned int /* version */);

        template<class Archive>
        void serialize(Archive & ar,
                       pinocchio::CollisionPair & collisionPair,
                       const unsigned int /* version */)
        {
            ar & make_nvp("base", base_object<
                std::pair<pinocchio::GeomIndex, pinocchio::GeomIndex> >(collisionPair));
        }

        template <class Archive>
        void serialize(Archive & ar,
                       pinocchio::GeometryModel & model,
                       const unsigned int /* version */)
        {
            ar & make_nvp("ngeoms", model.ngeoms);
            ar & make_nvp("geometryObjects", model.geometryObjects);
            ar & make_nvp("collisionPairs", model.collisionPairs);
        }
  } // namespace serialization
} // namespace boost

#endif // JIMINY_SERIALIZATION_TPP
