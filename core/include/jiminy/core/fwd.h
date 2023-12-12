#ifndef JIMINY_FORWARD_H
#define JIMINY_FORWARD_H

#include <cstdint>        // `int32_t`, `int64_t`, `uint32_t`, `uint64_t`, ...
#include <functional>     // `std::function`
#include <limits>         // `std::numeric_limits`
#include <map>            // `std::map`
#include <string>         // `std::string`
#include <unordered_map>  // `std::unordered_map`
#include <utility>        // `std::pair`
#include <vector>         // `std::vector`

#include "pinocchio/fwd.hpp"            // To avoid having to include it everywhere
#include "pinocchio/multibody/fwd.hpp"  // `pinocchio::JointIndex`, `pinocchio::FrameIndex`, ...

#include "pinocchio/container/aligned-vector.hpp"  // `pinocchio::container::aligned_vector`
#include "pinocchio/spatial/force.hpp"             // `pinocchio::Force`
#include "pinocchio/spatial/motion.hpp"            // `pinocchio::Motion`

#include <Eigen/Core>       // `Eigen::Matrix`, `Eigen::Dynamic`
#include <Eigen/StdVector>  // `Eigen::aligned_allocator`

#include <boost/variant.hpp>  // `boost::make_recursive_variant`


namespace jiminy
{
    // **************************************** Macros ***************************************** //

#define DISABLE_COPY(className)                  \
    className(const className & other) = delete; \
    className & operator=(const className & other) = delete;

#if defined _WIN32 || defined __CYGWIN__
// On Microsoft Windows, use dllimport and dllexport to tag symbols
#    define JIMINY_DLLIMPORT __declspec(dllimport)
#    define JIMINY_DLLEXPORT __declspec(dllexport)
#else
// On Linux, for GCC >= 4, tag symbols using GCC extension
#    define JIMINY_DLLIMPORT __attribute__((visibility("default")))
#    define JIMINY_DLLEXPORT __attribute__((visibility("default")))
#endif

// Define DLLAPI to import or export depending on whether one is building or using the library
#ifdef EXPORT_SYMBOLS
#    define JIMINY_DLLAPI JIMINY_DLLEXPORT
#else
#    define JIMINY_DLLAPI JIMINY_DLLIMPORT
#endif

    // ********************************** General definitions ********************************** //

    // "Standard" types
    using bool_t = bool;
    using char_t = char;
    using float32_t = float;
    using float64_t = double;

    template<typename K, typename M>
    using static_map_t = std::vector<std::pair<K, M>>;

    // Eigen types
    template<typename Scalar>
    using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar>
    using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename Scalar>
    using Matrix3X = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;

    using Matrix6Xd = Eigen::Matrix<float64_t, 6, Eigen::Dynamic>;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Size) \
    template<typename Scalar>           \
    using Vector##Size = Eigen::Matrix<Scalar, Size, 1>;

    EIGEN_MAKE_FIXED_TYPEDEFS(1)
    EIGEN_MAKE_FIXED_TYPEDEFS(2)
    EIGEN_MAKE_FIXED_TYPEDEFS(3)
    EIGEN_MAKE_FIXED_TYPEDEFS(6)

#undef EIGEN_MAKE_FIXED_TYPEDEFS

    using Vector6d = Vector6<float64_t>;

    template<typename K, typename M>
    using static_map_aligned_t =
        std::vector<std::pair<K, M>, Eigen::aligned_allocator<std::pair<K, M>>>;

    template<typename K, typename M>
    using map_aligned_t =
        std::map<K, M, std::less<K>, Eigen::aligned_allocator<std::pair<const K, M>>>;

    // Pinocchio types
    template<typename M>
    using vector_aligned_t = pinocchio::container::aligned_vector<M>;

    using MotionVector = vector_aligned_t<pinocchio::Motion>;
    using ForceVector = vector_aligned_t<pinocchio::Force>;

    enum class JointModelType : uint8_t
    {
        /// @brief Cylindrical joints are not available so far
        UNSUPPORTED = 0,
        LINEAR = 1,
        ROTARY = 2,
        /// @brief The configuration of unbounded rotary joints must be encoded as
        ///        `[cos(theta), sin(theta)]` instead of `theta` to prevent overflow.
        ROTARY_UNBOUNDED = 3,
        PLANAR = 4,
        TRANSLATION = 5,
        SPHERICAL = 6,
        FREE = 7
    };

    // ******************************* Constant of the universe ******************************** //

    // Define some aliases for convenience
    inline constexpr float64_t INF = std::numeric_limits<float64_t>::infinity();
    inline constexpr float64_t EPS = std::numeric_limits<float64_t>::epsilon();
    inline constexpr float64_t qNAN = std::numeric_limits<float64_t>::quiet_NaN();

    // ****************************** Jiminy-specific definitions ****************************** //

    // Error codes
    enum class hresult_t : int32_t
    {
        SUCCESS = 1,
        ERROR_GENERIC = -1,
        ERROR_BAD_INPUT = -2,
        ERROR_INIT_FAILED = -3
    };

    // Ground profile functors
    using HeightmapFunctor =
        std::function<std::pair<float64_t /*height*/, Eigen::Vector3d /*normal*/>(
            const Eigen::Vector3d & /*pos*/)>;

    // Flexible joints
    struct FlexibleJointData
    {
        // FIXME: Replace by default spaceship operator `<=>` starting from C++20.
        inline bool_t operator==(const FlexibleJointData & other) const noexcept
        {
            return (this->frameName == other.frameName && this->stiffness == other.stiffness &&
                    this->damping == other.damping && this->inertia == other.inertia);
        };

        std::string frameName;
        Eigen::Vector3d stiffness;
        Eigen::Vector3d damping;
        Eigen::Vector3d inertia;
    };

    using FlexibilityConfig = std::vector<FlexibleJointData>;

    using GenericConfig =
        std::unordered_map<std::string,
                           boost::make_recursive_variant<
                               bool_t,
                               uint32_t,
                               int32_t,
                               float64_t,
                               std::string,
                               Eigen::VectorXd,
                               Eigen::MatrixXd,
                               HeightmapFunctor,
                               std::vector<std::string>,
                               std::vector<Eigen::VectorXd>,
                               std::vector<Eigen::MatrixXd>,
                               FlexibilityConfig,
                               std::unordered_map<std::string, boost::recursive_variant_>>::type>;

    struct SensorDataTypeMap;
    using SensorsDataMap = std::unordered_map<std::string, SensorDataTypeMap>;
}  // namespace jiminy

#endif  // JIMINY_FORWARD_H
