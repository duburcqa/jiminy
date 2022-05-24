#ifndef JIMINY_SERIALIZATION_H
#define JIMINY_SERIALIZATION_H

#include "Serialization.tpp"

#include "hpp/fcl/shape/geometric_shapes.h"  // `hpp::fcl::ShapeBase`
#include "hpp/fcl/BVH/BVH_model.h"           // `hpp::fcl::BVHModel`

#include <boost/serialization/export.hpp>    // `BOOST_CLASS_EXPORT`

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
