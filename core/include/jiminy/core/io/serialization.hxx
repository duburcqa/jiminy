#ifndef JIMINY_SERIALIZATION_HXX
#define JIMINY_SERIALIZATION_HXX

#include <sstream>

#include "pinocchio/multibody/fcl.hpp"           // `pinocchio::CollisionPair`
#include "pinocchio/multibody/geometry.hpp"      // `pinocchio::GeometryModel`
#include "pinocchio/serialization/model.hpp"     // `serialize<pinocchio::Model>`
#include "pinocchio/serialization/geometry.hpp"  // `serialize<pinocchio::CollisionPair>`

#define HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION
#include "hpp/fcl/serialization/geometric_shapes.h"  // `serialize<hpp::fcl::ShapeBase>`
#include "hpp/fcl/serialization/convex.h"            // `serialize<hpp::fcl::ConvexBase>`
#undef HPP_FCL_SKIP_EIGEN_BOOST_SERIALIZATION

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


namespace jiminy
{
    template<typename T>
    std::string saveToBinary(const T & obj)
    {
        std::ostringstream os;
        {
            boost::archive::binary_oarchive oa(os);
            oa << obj;
            return os.str();
        }
    }

    template<typename T>
    void loadFromBinary(T & obj, const std::string & str)
    {
        std::istringstream is(str);
        {
            boost::archive::binary_iarchive ia(is);
            ia >> obj;
        }
    }
}


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
    serialize(Archive & ar, pinocchio::GeometryObject & geom, const unsigned int /* version */);

    template<class Archive>
    void
    serialize(Archive & ar, pinocchio::GeometryModel & model, const unsigned int /* version */)
    {
        ar & make_nvp("ngeoms", model.ngeoms);
        ar & make_nvp("geometryObjects", model.geometryObjects);
        ar & make_nvp("collisionPairs", model.collisionPairs);
    }
}

#endif  // JIMINY_SERIALIZATION_HXX
