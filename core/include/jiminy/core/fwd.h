#ifndef JIMINY_FORWARD_H
#define JIMINY_FORWARD_H

#include <string_view>    // `std::string_view`
#include <cstdint>        // `int32_t`, `int64_t`, `uint32_t`, `uint64_t`, ...
#include <functional>     // `std::function`
#include <limits>         // `std::numeric_limits`
#include <map>            // `std::map`
#include <string>         // `std::string`
#include <sstream>        // `std::ostringstream`
#include <unordered_map>  // `std::unordered_map`
#include <utility>        // `std::pair`
#include <vector>         // `std::vector`
#include <stdexcept>      // `std::runtime_error`, `std::logic_error`

#include "pinocchio/fwd.hpp"            // To avoid having to include it everywhere
#include "pinocchio/multibody/fwd.hpp"  // `pinocchio::JointIndex`, `pinocchio::FrameIndex`, ...

#include "pinocchio/container/aligned-vector.hpp"  // `pinocchio::container::aligned_vector`
#include "pinocchio/spatial/force.hpp"             // `pinocchio::Force`
#include "pinocchio/spatial/motion.hpp"            // `pinocchio::Motion`

#include <Eigen/Core>  // `Eigen::Matrix`, `Eigen::Dynamic`, `Eigen::IOFormat`, `Eigen::FullPrecision`
#include <Eigen/StdVector>    // `Eigen::aligned_allocator`
#include <boost/variant.hpp>  // `boost::make_recursive_variant`

#include "jiminy/core/constants.h"
#include "jiminy/core/macros.h"
#include "jiminy/core/traits.h"


namespace jiminy
{
    // ********************************** General declarations ********************************* //

    template<typename K, typename M>
    using static_map_t = std::vector<std::pair<K, M>>;

    // Eigen types
    template<typename Scalar>
    using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    template<typename Scalar>
    using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename Scalar>
    using Matrix3X = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;

    using Matrix6Xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Size) \
    template<typename Scalar>           \
    using Vector##Size = Eigen::Matrix<Scalar, Size, 1>;

    EIGEN_MAKE_FIXED_TYPEDEFS(1)
    EIGEN_MAKE_FIXED_TYPEDEFS(2)
    EIGEN_MAKE_FIXED_TYPEDEFS(3)
    EIGEN_MAKE_FIXED_TYPEDEFS(6)

#undef EIGEN_MAKE_FIXED_TYPEDEFS

    using Vector6d = Vector6<double>;

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

    // ****************************** Jiminy-specific declarations ***************************** //

    // Exceptions
    class jiminy_exception : public std::exception
    {
    public:
        using std::exception::exception;
    };

    class not_initialized : public jiminy_exception, public std::logic_error
    {
    public:
        using std::logic_error::logic_error;
        using std::logic_error::logic_error::what;
    };

    class initialization_failed : public jiminy_exception, public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
        using std::runtime_error::runtime_error::what;
    };

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
        std::function<std::pair<double /*height*/, Eigen::Vector3d /*normal*/>(
            const Eigen::Vector3d & /*pos*/)>;

    // Flexible joints
    struct FlexibleJointData
    {
        // FIXME: Replace by default spaceship operator `<=>` when moving to C++20.
        inline bool operator==(const FlexibleJointData & other) const noexcept
        {
            return (this->frameName == other.frameName && this->stiffness == other.stiffness &&
                    this->damping == other.damping && this->inertia == other.inertia);
        };

        std::string frameName{};
        Eigen::Vector3d stiffness{};
        Eigen::Vector3d damping{};
        Eigen::Vector3d inertia{};
    };

    using FlexibilityConfig = std::vector<FlexibleJointData>;

    using GenericConfig =
        std::unordered_map<std::string,
                           boost::make_recursive_variant<
                               bool,
                               uint32_t,
                               int32_t,
                               double,
                               std::string,
                               Eigen::VectorXd,
                               Eigen::MatrixXd,
                               HeightmapFunctor,
                               std::vector<std::string>,
                               std::vector<Eigen::VectorXd>,
                               std::vector<Eigen::MatrixXd>,
                               FlexibilityConfig,
                               std::unordered_map<std::string, boost::recursive_variant_>>::type>;

    // Generic utilities used everywhere
    template<typename... Args>
    std::string toString(Args &&... args)
    {
        std::ostringstream sstr;
        auto format = [](auto && var)
        {
            if constexpr (is_eigen_v<decltype(var)>)
            {
                static const Eigen::IOFormat k_heavy_fmt(
                    Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
                return var.format(k_heavy_fmt);
            }
            return var;
        };
        ((sstr << format(std::forward<Args>(args))), ...);
        return sstr.str();
    }
}

#endif  // JIMINY_FORWARD_H
