///////////////////////////////////////////////////////////////////////////////
/// \brief    Contains types used in Jiminy.
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_TYPES_H
#define JIMINY_TYPES_H

#include <string>
#include <vector>
#include <unordered_map>

#include "pinocchio/fwd.hpp"          // To avoid having to include it everywhere
#include "pinocchio/spatial/fwd.hpp"  // `Pinocchio::Force`, `Pinocchio::Motion`

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// std::optional is not vailable for gcc<=7.3, so using boost instead
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <boost/functional/hash.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/tag.hpp>


// `pinocchio::container::aligned_vector`
namespace pinocchio
{
    namespace container
    {
        template<typename T> struct aligned_vector;
    }
}


namespace jiminy
{
    // ******************* General definitions *******************

    // "Standard" types
    using bool_t = bool;
    using char_t = char;
    using float32_t = float;
    using float64_t = double;

    template<typename K, typename M>
    using static_map_t = std::vector<std::pair<K, M> >;

    // Eigen types
    using matrixN_t = Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic>;
    using matrix2_t = Eigen::Matrix<float64_t, 2, 2>;
    using matrix3_t = Eigen::Matrix<float64_t, 3, 3>;
    using vectorN_t = Eigen::Matrix<float64_t, Eigen::Dynamic, 1>;
    using vector3_t = Eigen::Matrix<float64_t, 3, 1>;
    using vector6_t = Eigen::Matrix<float64_t, 6, 1>;
    using rowN_t = Eigen::Matrix<float64_t, 1, Eigen::Dynamic>;

    using constMatrixBlock_t = Eigen::Block<matrixN_t const, Eigen::Dynamic, Eigen::Dynamic> const;
    using constVectorBlock_t = Eigen::VectorBlock<vectorN_t const, Eigen::Dynamic> const;

    using quaternion_t = Eigen::Quaternion<float64_t>;

    // Pinocchio types
    using motionVector_t = pinocchio::container::aligned_vector<pinocchio::Motion>;
    using forceVector_t = pinocchio::container::aligned_vector<pinocchio::Force>;

    // *************** Constant of the universe ******************

    // Define some aliases for convenience
    float64_t const INF = std::numeric_limits<float64_t>::infinity();
    float64_t const EPS = std::numeric_limits<float64_t>::epsilon();
    float64_t const qNAN = std::numeric_limits<float64_t>::quiet_NaN();

    // *************** Jiminy-specific definitions ***************

    // Error codes
    enum class hresult_t : int32_t
    {
        SUCCESS = 1,
        ERROR_GENERIC = -1,
        ERROR_BAD_INPUT = -2,
        ERROR_INIT_FAILED = -3
    };

    // Pinocchio joint types
    enum class joint_t : uint8_t
    {
        // CYLINDRICAL are not available so far

        NONE = 0,
        LINEAR = 1,
        ROTARY = 2,
        ROTARY_UNBOUNDED = 3,  // Must be treated separately because the position is encoded using [cos(theta), sin(theta)] instead of theta
        PLANAR = 4,
        TRANSLATION = 5,
        SPHERICAL = 6,
        FREE = 7
    };

    /* Ground profile signature.
       Note that it is impossible to use function pointer since it does not support functors. */
    using heatMapFunctor_t = std::function<std::pair<float64_t, vector3_t>(vector3_t const & /*pos*/)>;

    // Flexible joints
    struct flexibleJointData_t
    {
        std::string frameName;
        vectorN_t stiffness;
        vectorN_t damping;

        flexibleJointData_t(void) :
        frameName(),
        stiffness(),
        damping()
        {
            // Empty.
        };

        flexibleJointData_t(std::string const & frameNameIn,
                            vectorN_t   const & stiffnessIn,
                            vectorN_t   const & dampingIn) :
        frameName(frameNameIn),
        stiffness(stiffnessIn),
        damping(dampingIn)
        {
            // Empty.
        };

        inline bool_t operator==(flexibleJointData_t const & other) const
        {
            return (this->frameName == other.frameName
                 && this->stiffness == other.stiffness
                 && this->damping == other.damping);
        };
    };

    using flexibilityConfig_t = std::vector<flexibleJointData_t>;

    // Configuration/option holder
    using configField_t = boost::make_recursive_variant<
        bool_t, uint32_t, int32_t, float64_t, std::string, vectorN_t, matrixN_t, heatMapFunctor_t,
        std::vector<std::string>, std::vector<vectorN_t>, std::vector<matrixN_t>, flexibilityConfig_t,
        std::unordered_map<std::string, boost::recursive_variant_>
    >::type;

    using configHolder_t = std::unordered_map<std::string, configField_t>;

    // Sensor data holder
    struct sensorDataTypePair_t
    {
        // Disable the copy of the class
        sensorDataTypePair_t(sensorDataTypePair_t const & sensorDataPairIn) = delete;
        sensorDataTypePair_t & operator = (sensorDataTypePair_t const & other) = delete;

        sensorDataTypePair_t(std::string                 const & nameIn,
                             int32_t                     const & idIn,
                             Eigen::Ref<vectorN_t const> const & valueIn) :
        name(nameIn),
        idx(idIn),
        value(valueIn)
        {
            // Empty on purpose.
        };

        ~sensorDataTypePair_t(void) = default;

        sensorDataTypePair_t(sensorDataTypePair_t && other) :
        name(other.name),
        idx(other.idx),
        value(other.value)
        {
            // Empty on purpose.
        };

        std::string name;
        int32_t idx;
        Eigen::Ref<vectorN_t const> value;
    };

    using namespace boost::multi_index;
    struct IndexByIdx {};
    struct IndexByName {};
    using sensorDataTypeMapImpl_t = multi_index_container<
        sensorDataTypePair_t,
        indexed_by<
            ordered_unique<
                tag<IndexByIdx>,
                member<sensorDataTypePair_t, int32_t, &sensorDataTypePair_t::idx>,
                std::less<int32_t> // Ordering by ascending order
            >,
            hashed_unique<
                tag<IndexByName>,
                member<sensorDataTypePair_t, std::string, &sensorDataTypePair_t::name>
            >
        >
    >;
    struct sensorDataTypeMap_t : public sensorDataTypeMapImpl_t
    {
    public:
        sensorDataTypeMap_t(boost::optional<std::reference_wrapper<matrixN_t const> > sharedData = boost::none) :
        sensorDataTypeMapImpl_t(),
        sharedData_(sharedData)
        {
            // Empty on purpose.
        }

        matrixN_t const & getAll(void) const
        {
            /* Use static shared buffer if no shared memory available.
               It is useful if the sensors data is not contiguous in the first place,
               which is likely to be the case when allocated from Python, or when
               re-generating sensor data from log files. */
            static matrixN_t sharedData;

            if (sharedData_.is_initialized())
            {
                /* Return shared memory directly it is up to the sure to make sure
                   that it is actually up-to-date. */
                return sharedData_.value().get();
            }
            else
            {
                // Resize internal buffer if needed
                if (size() != size_type(sharedData.rows()))
                {
                    sharedData.resize(size(), this->begin()->value.size());
                }

                // Set internal buffer by copying sensor data sequentially
                for (auto const & sensor : *this)
                {
                    sharedData.row(sensor.idx) = sensor.value;
                }

                return sharedData;
            }
        }

    private:
        boost::optional<std::reference_wrapper<matrixN_t const> > sharedData_;
    };

    using sensorsDataMap_t = std::unordered_map<std::string, sensorDataTypeMap_t>;

    // System force functors
    using forceProfileFunctor_t = std::function<pinocchio::Force(float64_t const & /*t*/,
                                                                 vectorN_t const & /*q*/,
                                                                 vectorN_t const & /*v*/)>;
    using forceCouplingFunctor_t = std::function<pinocchio::Force(float64_t const & /*t*/,
                                                                  vectorN_t const & /*q_1*/,
                                                                  vectorN_t const & /*v_1*/,
                                                                  vectorN_t const & /*q_2*/,
                                                                  vectorN_t const & /*v_2*/)>;

    // System callback functor
    using callbackFunctor_t = std::function<bool_t(float64_t const & /*t*/,
                                                   vectorN_t const & /*q*/,
                                                   vectorN_t const & /*v*/)>;
}

#endif  // JIMINY_TYPES_H
