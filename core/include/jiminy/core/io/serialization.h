#ifndef JIMINY_SERIALIZATION_H
#define JIMINY_SERIALIZATION_H

#include <optional>  // `std::optional`
#include <string>    // `std::string`
#include <vector>    // `std::vector`
#include <memory>    // `std::shared_ptr`

#include "jiminy/core/fwd.h"


// Forward declarations
namespace pinocchio
{
    struct GeometryObject;
    struct GeometryModel;
}

namespace jiminy
{
    class Model;
    class Robot;

    class stateful_binary_iarchive;
    class stateful_binary_oarchive;
}

/* Jiminy-specific API for saving and loading data to binary format.
   Internally, it leverages `boost::serialization`, but unlike the classic interface `serialize`,
   it can be specified to pass option arguments, which allows for providing opt-in storage
   optimizations and recovery information when loading partially corrupted data. */
namespace jiminy
{
    template<typename T>
    std::string saveToBinary(const T & obj);

    template<typename T>
    void loadFromBinary(T & obj, const std::string & data);

    std::string saveToBinary(const std::shared_ptr<const jiminy::Robot> & robot,
                             bool isPersistent = true);

    void loadFromBinary(std::shared_ptr<jiminy::Robot> & robot,
                        const std::string & data,
                        const std::optional<std::string> & meshPathDir = std::nullopt,
                        const std::vector<std::string> & meshPackageDirs = {});
}

/* Partial specialization of `boost::serialization` API to enable serialization of complex classes.
   Unlike `loadFromBinary`, `saveToBinary`, this API does not expose mechanics to customize its
   internal, so the most conservation assumptions are made, eventually impeding performance. */
namespace boost::serialization
{
    // *************************************** pinocchio *************************************** //

    template<class Archive>
    void load_construct_data(
        Archive & ar, pinocchio::GeometryObject * geomPtr, const unsigned int version);

    template<class Archive>
    void serialize(Archive & ar, pinocchio::GeometryObject & geom, const unsigned int version);

    template<class Archive>
    void serialize(Archive & ar, pinocchio::GeometryModel & model, const unsigned int version);

    // ***************************************** jiminy **************************************** //

    template<class Archive>
    void load_construct_data(Archive & ar, jiminy::Model * modelPtr, const unsigned int version);

    template<class Archive>
    void serialize(Archive & ar, jiminy::Model & model, const unsigned int version);

    template<class Archive>
    void
    save_construct_data(Archive & ar, const jiminy::Robot * robotPtr, const unsigned int version);

    template<class Archive>
    void load_construct_data(Archive & ar, jiminy::Robot * robotPtr, const unsigned int version);

    template<class Archive>
    void serialize(Archive & ar, jiminy::Robot & robot, const unsigned int version);
}

#include "jiminy/core/io/serialization.hxx"

#endif  // JIMINY_SERIALIZATION_H
