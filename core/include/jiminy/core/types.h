#ifndef JIMINY_TYPES_H
#define JIMINY_TYPES_H

#include <string>
#include <vector>
#include <unordered_map>

#include "pinocchio/fwd.hpp"            // To avoid having to include it everywhere
#include "pinocchio/multibody/fwd.hpp"  // `pinocchio::Model::...Index`
#include "pinocchio/spatial/fwd.hpp"    // `Pinocchio::Force`, `Pinocchio::Motion`

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <boost/variant.hpp>
#include <boost/functional/hash.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/tag.hpp>


// `pinocchio::container::aligned_vector`
namespace pinocchio::container
{
    template<typename T>
    struct aligned_vector;
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
    using static_map_t = std::vector<std::pair<K, M>>;

    template<typename K, typename M>
    using static_map_aligned_t =
        std::vector<std::pair<K, M>, Eigen::aligned_allocator<std::pair<K, M>>>;

    template<typename K, typename M>
    using map_aligned_t =
        std::map<K, M, std::less<K>, Eigen::aligned_allocator<std::pair<const K, M>>>;

    template<typename M>
    using vector_aligned_t = pinocchio::container::aligned_vector<M>;

    // Eigen types
    using Matrix6Xd = Eigen::Matrix<float64_t, 6, Eigen::Dynamic>;
    using Vector6d = Eigen::Matrix<float64_t, 6, 1>;

    using quaternion_t = Eigen::Quaternion<float64_t>;

    // Pinocchio types
    using motionVector_t = vector_aligned_t<pinocchio::Motion>;
    using forceVector_t = vector_aligned_t<pinocchio::Force>;
    using jointIndex_t = pinocchio::JointIndex;
    using frameIndex_t = pinocchio::FrameIndex;
    using geomIndex_t = pinocchio::GeomIndex;
    using pairIndex_t = pinocchio::PairIndex;

    // *************** Constant of the universe ******************

    // Define some aliases for convenience
    const float64_t INF = std::numeric_limits<float64_t>::infinity();
    const float64_t EPS = std::numeric_limits<float64_t>::epsilon();
    const float64_t qNAN = std::numeric_limits<float64_t>::quiet_NaN();

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
        /// \details Must be distinguished from ROTARY because the position is not encoded as theta
        ///          but rather [cos(theta), sin(theta)] to avoid potential overflow.
        ROTARY_UNBOUNDED = 3,
        PLANAR = 4,
        TRANSLATION = 5,
        SPHERICAL = 6,
        FREE = 7
    };

    /* Ground profile signature.
       Note that it is impossible to use function pointer since it does not support functors. */
    using heightmapFunctor_t =
        std::function<std::pair<float64_t, Eigen::Vector3d>(const Eigen::Vector3d & /*pos*/)>;

    // Flexible joints
    struct flexibleJointData_t
    {
    public:
        inline bool_t operator==(const flexibleJointData_t & other) const
        {
            return (this->frameName == other.frameName && this->stiffness == other.stiffness &&
                    this->damping == other.damping && this->inertia == other.inertia);
        };

    public:
        std::string frameName;
        Eigen::Vector3d stiffness;
        Eigen::Vector3d damping;
        Eigen::Vector3d inertia;
    };

    using flexibilityConfig_t = std::vector<flexibleJointData_t>;

    // Configuration/option holder
    using configField_t = boost::make_recursive_variant<
        bool_t,
        uint32_t,
        int32_t,
        float64_t,
        std::string,
        Eigen::VectorXd,
        Eigen::MatrixXd,
        heightmapFunctor_t,
        std::vector<std::string>,
        std::vector<Eigen::VectorXd>,
        std::vector<Eigen::MatrixXd>,
        flexibilityConfig_t,
        std::unordered_map<std::string, boost::recursive_variant_>>::type;

    using configHolder_t = std::unordered_map<std::string, configField_t>;

    // Sensor data holder
    struct sensorDataTypePair_t
    {
    public:
        sensorDataTypePair_t(const std::string & nameIn,
                             const Eigen::Index & idIn,
                             const Eigen::Ref<const Eigen::VectorXd> & valueIn) :
        name(nameIn),
        idx(idIn),
        value(valueIn)
        {
        }

    public:
        std::string name;
        Eigen::Index idx;
        Eigen::Ref<const Eigen::VectorXd> value;
    };

    using namespace boost::multi_index;
    struct IndexByIdx
    {
    };
    struct IndexByName
    {
    };
    using sensorDataTypeMapImpl_t = multi_index_container<
        sensorDataTypePair_t,
        indexed_by<
            ordered_unique<tag<IndexByIdx>,
                           member<sensorDataTypePair_t, Eigen::Index, &sensorDataTypePair_t::idx>,
                           std::less<Eigen::Index>  // Ordering by ascending order
                           >,
            hashed_unique<tag<IndexByName>,
                          member<sensorDataTypePair_t, std::string, &sensorDataTypePair_t::name>>>>;
    struct sensorDataTypeMap_t : public sensorDataTypeMapImpl_t
    {
    public:
        sensorDataTypeMap_t(
            std::optional<std::reference_wrapper<const Eigen::MatrixXd>> sharedData =
                std::nullopt) :
        sensorDataTypeMapImpl_t(),
        sharedDataRef_(sharedData)
        {
        }

        inline const Eigen::MatrixXd & getAll() const
        {
            if (sharedDataRef_)
            {
                /* Return shared memory directly. It is up to the user to make sure that it is
                   actually up-to-date. */
                assert(size() == static_cast<std::size_t>(sharedDataRef_->get().cols()) &&
                       "Shared data inconsistent with sensors.");
                return sharedDataRef_->get();
            }
            else
            {
                // Get sensors data size
                Eigen::Index dataSize = 0;
                if (size() > 0)
                {
                    dataSize = this->begin()->value.size();
                }

                // Resize internal buffer if needed
                sharedData_.resize(dataSize, static_cast<Eigen::Index>(size()));

                // Set internal buffer by copying sensor data sequentially
                for (const auto & sensor : *this)
                {
                    assert(sensor.value.size() == dataSize &&
                           "Cannot get all data at once for heterogeneous sensors.");
                    sharedData_.col(sensor.idx) = sensor.value;
                }

                return sharedData_;
            }
        }

    private:
        std::optional<std::reference_wrapper<const Eigen::MatrixXd>> sharedDataRef_;
        /* Internal buffer if no shared memory available.
           It is useful if the sensors data is not contiguous in the first place, which is likely
           to be the case when allocated from Python, or when re-generating sensor data from log
           files. */
        mutable Eigen::MatrixXd sharedData_{};
    };

    using sensorsDataMap_t = std::unordered_map<std::string, sensorDataTypeMap_t>;

    // System force functors
    using forceProfileFunctor_t = std::function<pinocchio::Force(
        const float64_t & /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;
    using forceCouplingFunctor_t =
        std::function<pinocchio::Force(const float64_t & /*t*/,
                                       const Eigen::VectorXd & /*q_1*/,
                                       const Eigen::VectorXd & /*v_1*/,
                                       const Eigen::VectorXd & /*q_2*/,
                                       const Eigen::VectorXd & /*v_2*/)>;

    // System callback functor
    using callbackFunctor_t = std::function<bool_t(
        const float64_t & /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;

    // Log data type
    struct logData_t
    {
        int32_t version;
        float64_t timeUnit;
        Eigen::Matrix<int64_t, Eigen::Dynamic, 1> timestamps;
        static_map_t<std::string, std::string> constants;
        std::vector<std::string> fieldnames;
        Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> intData;
        Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> floatData;
    };
}

#endif  // JIMINY_TYPES_H
