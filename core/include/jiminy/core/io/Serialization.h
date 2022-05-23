
#ifndef JIMINY_SERIALIZATION_H
#define JIMINY_SERIALIZATION_H

#include "pinocchio/multibody/fcl.hpp"          // `pinocchio::CollisionPair`
#include "pinocchio/multibody/geometry.hpp"     // `pinocchio::GeometryModel`
#include "pinocchio/serialization/archive.hpp"  // `pinocchio::serialization::saveToString`
#include "pinocchio/serialization/model.hpp"    // `serialize<pinocchio::Model>`

#define HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION
#include "hpp/fcl/serialization/collision_object.h"  // `serialize<hpp::fcl::CollisionGeometry>`
#include "hpp/fcl/serialization/geometric_shapes.h"  // `serialize<hpp::fcl::ShapeBase>`
#include "hpp/fcl/serialization/BVH_model.h"         // `serialize<hpp::fcl::BVHModel>`
#include <boost/serialization/export.hpp>            // `BOOST_CLASS_EXPORT`


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
                ar & BOOST_SERIALIZATION_NVP(rows);
            }
            if (PlainObjectBase::ColsAtCompileTime == Eigen::Dynamic)
            {
                ar & BOOST_SERIALIZATION_NVP(cols);
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

BOOST_CLASS_EXPORT_KEY(hpp::fcl::TriangleP)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::Sphere)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::Box)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::Capsule)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::Cone)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::Cylinder)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::Halfspace)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::Plane)
// BOOST_CLASS_EXPORT_KEY(hpp::fcl::ConvexBase)  // Convex are not serializable for now
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::AABB>)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::OBB>)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::RSS>)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::OBBRSS>)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::kIOS>)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::KDOP<16> >)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::KDOP<18> >)
BOOST_CLASS_EXPORT_KEY(hpp::fcl::BVHModel<hpp::fcl::KDOP<24> >)

#endif // JIMINY_SERIALIZATION_H
