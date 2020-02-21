///////////////////////////////////////////////////////////////////////////////
/// \brief    Contains types used in the optimal module.
///////////////////////////////////////////////////////////////////////////////

#ifndef WDC_OPTIMAL_TYPES_H
#define WDC_OPTIMAL_TYPES_H

#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/variant.hpp>
#include <boost/functional/hash.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/tag.hpp>


namespace jiminy
{
    // "Standard" types
    typedef bool   bool_t;
    typedef char   char_t;
    typedef float  float32_t;
    typedef double float64_t;
    typedef char_t const* const const_cstr_t;

    // Math types
    typedef Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic> matrixN_t;
    typedef Eigen::Matrix<float64_t, 3, 3>                           matrix3_t;
    typedef Eigen::Matrix<float64_t, Eigen::Dynamic, 1>              vectorN_t;
    typedef Eigen::Matrix<float64_t, 3, 1>                           vector3_t;
    typedef Eigen::Matrix<float64_t, 6, 1>                           vector6_t;
    typedef Eigen::Matrix<float64_t, 1, Eigen::Dynamic>              rowN_t;

    typedef Eigen::Block<matrixN_t const, Eigen::Dynamic, Eigen::Dynamic> constBlockXpr;
    typedef Eigen::Block<matrixN_t, Eigen::Dynamic, Eigen::Dynamic> blockXpr;

    typedef Eigen::Quaternion<float64_t> quaternion_t;

    float64_t const INF = std::numeric_limits<float64_t>::infinity();
    float64_t const EPS = std::numeric_limits<float64_t>::epsilon();

    // Jiminy-specific type
    enum class result_t : int32_t
    {
        SUCCESS = 1,
        ERROR_GENERIC = -1,
        ERROR_BAD_INPUT = -2,
        ERROR_INIT_FAILED = -3
    };

    typedef std::function<std::pair<float64_t, vector3_t>(vector3_t const & /*pos*/)> heatMapFunctor_t; // Impossible to use function pointer since it does not support functors

    struct flexibleJointData_t
    {
        std::string jointName;
        vectorN_t stiffness;
        vectorN_t damping;

        flexibleJointData_t(void) :
        jointName(),
        stiffness(),
        damping()
        {
            // Empty.
        };

        flexibleJointData_t(std::string const & jointNameIn,
                            vectorN_t   const & stiffnessIn,
                            vectorN_t   const & dampingIn) :
        jointName(jointNameIn),
        stiffness(stiffnessIn),
        damping(dampingIn)
        {
            // Empty.
        };

        inline bool operator==(flexibleJointData_t const & other) const
        {
            return (this->jointName == other.jointName
                 && this->stiffness == other.stiffness
                 && this->damping == other.damping);
        };
    };
    typedef std::vector<flexibleJointData_t> flexibilityConfig_t;

    typedef boost::make_recursive_variant<bool_t, uint32_t, int32_t, float64_t, std::string, vectorN_t, matrixN_t,
                                          std::vector<std::string>, std::vector<vectorN_t>, std::vector<matrixN_t>,
                                          flexibilityConfig_t, heatMapFunctor_t,
                                          std::unordered_map<std::string, boost::recursive_variant_> >::type configField_t;
    typedef std::unordered_map<std::string, configField_t> configHolder_t;

    using namespace boost::multi_index;
    struct sensorDataTypePair_t {
        // Disable the copy of the class
        sensorDataTypePair_t(sensorDataTypePair_t const & sensorDataPairIn) = delete;
        sensorDataTypePair_t & operator = (sensorDataTypePair_t const & other) = delete;

        sensorDataTypePair_t(std::string const & nameIn,
                             uint32_t    const & idIn,
                             vectorN_t   const * valueIn) :
        name(nameIn),
        id(idIn),
        value(valueIn)
        {
            // Empty.
        };

        ~sensorDataTypePair_t(void) = default;

        sensorDataTypePair_t(sensorDataTypePair_t && other) :
        name(other.name),
        id(other.id),
        value(other.value)
        {
            // Empty.
        };

        std::string name;
        uint32_t id;
        vectorN_t const * value;
    };
    struct IndexByName {};
    struct IndexById {};
    typedef multi_index_container<
        sensorDataTypePair_t,
        indexed_by<
            ordered_unique<
                tag<IndexById>,
                member<sensorDataTypePair_t, uint32_t, &sensorDataTypePair_t::id>,
                std::less<uint32_t> // Ordering by ascending order
            >,
            hashed_unique<
                tag<IndexByName>,
                member<sensorDataTypePair_t, std::string, &sensorDataTypePair_t::name>
            >
        >
    > sensorDataTypeMap_t;
    typedef std::unordered_map<std::string, sensorDataTypeMap_t> sensorsDataMap_t;
}

#endif  // WDC_OPTIMAL_TYPES_H
