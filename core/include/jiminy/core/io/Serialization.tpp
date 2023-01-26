#ifndef JIMINY_SERIALIZATION_TPP
#define JIMINY_SERIALIZATION_TPP

#include <sstream>

#include "pinocchio/multibody/fcl.hpp"           // `pinocchio::CollisionPair`
#include "pinocchio/multibody/geometry.hpp"      // `pinocchio::GeometryModel`
#include "pinocchio/serialization/model.hpp"     // `serialize<pinocchio::Model>`
#include "pinocchio/serialization/geometry.hpp"  // `serialize<pinocchio::CollisionPair>`

#include "hpp/fcl/shape/convex.h"                    // `serialize<hpp::fcl::Convex>`
#define HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION
#include "hpp/fcl/serialization/geometric_shapes.h"  // `serialize<hpp::fcl::ShapeBase>`
#undef HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


namespace jiminy
{
    template<typename T>
    std::string saveToBinary(T const & obj)
    {
        std::ostringstream os;
        {
            boost::archive::binary_oarchive oa(os);
            oa << obj;
            return os.str();
        }
    }

    template<typename T>
    void loadFromBinary(T & obj, std::string const & str)
    {
        std::istringstream is(str);
        {
            boost::archive::binary_iarchive ia(is);
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
        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Sphere * spherePtr,
                                 unsigned int const /* version */)
        {
            ::new(spherePtr) hpp::fcl::Sphere(0.0);
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::TriangleP * trianglePPtr,
                                 unsigned int const /* version */)
        {
            ::new(trianglePPtr) hpp::fcl::TriangleP(
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Capsule * capsulePtr,
                                 unsigned int const /* version */)
        {
            ::new(capsulePtr) hpp::fcl::Capsule(0.0, 0.0);
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Cone * conePtr,
                                 unsigned int const /* version */)
        {
            ::new(conePtr) hpp::fcl::Cone(0.0, 0.0);
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 hpp::fcl::Cylinder * cylinderPtr,
                                 unsigned int const /* version */)
        {
            ::new(cylinderPtr) hpp::fcl::Cylinder(0.0, 0.0);
        }

        template<class Archive, typename PolygonT>
        inline void save_construct_data(Archive & ar,
                                        hpp::fcl::Convex<PolygonT> const * convexPtr,
                                        unsigned int const /* version */)
        {
            ar & make_nvp("num_points", convexPtr->num_points);
            ar & make_nvp("num_polygons", convexPtr->num_polygons);
            ar & make_nvp("points", make_array(convexPtr->points, convexPtr->num_points));
            ar & make_nvp("polygons", make_array(convexPtr->polygons, convexPtr->num_polygons));
        }

        template<class Archive, typename PolygonT>
        inline void load_construct_data(Archive & ar,
                                        hpp::fcl::Convex<PolygonT> * convexPtr,
                                        unsigned int const /* version */)
        {
            int numPoints, numPolygons;
            ar & make_nvp("num_points", numPoints);
            ar & make_nvp("num_polygons", numPolygons);
            hpp::fcl::Vec3f * points = new hpp::fcl::Vec3f[numPoints];
            PolygonT * polygons = new PolygonT[numPolygons];
            ar & make_nvp("points", make_array(points, numPoints));
            ar & make_nvp("polygons", make_array(polygons, numPolygons));
            ::new(convexPtr) hpp::fcl::Convex<PolygonT>(
                true, points, numPoints, polygons, numPolygons);
        }

        template <class Archive, typename PolygonT>
        void serialize(Archive & ar,
                       hpp::fcl::Convex<PolygonT> & convex,
                       unsigned int const /* version */)
        {
            ar & make_nvp(BOOST_PP_STRINGIZE(hpp::fcl::ShapeBase),
                boost::serialization::base_object<hpp::fcl::ShapeBase>(convex));
        }

        template<class Archive>
        void load_construct_data(Archive & /* ar */,
                                 pinocchio::GeometryObject * geomPtr,
                                 unsigned int const /* version */)
        {
            ::new(geomPtr) pinocchio::GeometryObject(
                "", 0, 0, {nullptr}, pinocchio::SE3::Identity());
        }

        template<class Archive>
        void serialize(Archive & ar,
                       pinocchio::GeometryObject & geom,
                       unsigned int const /* version */);

        template <class Archive>
        void serialize(Archive & ar,
                       pinocchio::GeometryModel & model,
                       unsigned int const /* version */)
        {
            ar & make_nvp("ngeoms", model.ngeoms);
            ar & make_nvp("geometryObjects", model.geometryObjects);
            ar & make_nvp("collisionPairs", model.collisionPairs);
        }
  } // namespace serialization
} // namespace boost

#endif // JIMINY_SERIALIZATION_TPP
