
#ifndef JIMINY_TRAITS_H
#define JIMINY_TRAITS_H

#include <map>            // `std::map`
#include <unordered_map>  // `std::unordered_map`
#include <type_traits>  // `std::enable_if_t`, `std::decay_t`, `std::add_pointer_t`, `std::true_type`, `std::false_type`
#include <utility>  // `std::declval`
#include <vector>   // `std::vector`

#include "pinocchio/multibody/joint/fwd.hpp"  // `pinocchio::JointModel ## type ## Tpl`, `pinocchio::JointData ## type ## Tpl`

#include <Eigen/Core>  // `Eigen::Matrix`, `Eigen::Ref`, `Eigen::VectorBlock`, `Eigen::Block`


// Note that multiple identical forward declarations are not an error, so no big deal if future
// pinocchio versions start to forward declare mimic joints.
namespace pinocchio
{
    template<class JointModel>
    struct JointModelMimic;

    template<class JointData>
    struct JointDataMimic;
}

namespace jiminy
{
    // ******************************** is_contiguous_container ******************************** //

    namespace internal
    {
        template<class...>
        struct voider
        {
            using type = void;
        };

        template<class... Ts>
        using void_t = typename voider<Ts...>::type;

        template<template<class...> class, class, class...>
        struct canApplyImply : std::false_type
        {
        };

        template<template<class...> class Z, class... Ts>
        struct canApplyImply<Z, void_t<Z<Ts...>>, Ts...> : std::true_type
        {
        };

        template<template<class...> class Z, class... Ts>
        using canApply = canApplyImply<Z, void, Ts...>;

        template<class T>
        using sizeFunType = decltype(std::declval<T>().size());

        template<class T>
        using hasSize = canApply<sizeFunType, T>;

        template<class T, class I = std::size_t>
        using rawIndexFunType = decltype(std::declval<T>().data()[std::declval<I>()]);

        template<class T, class I = std::size_t>
        using isContiguousIndexable = canApply<rawIndexFunType, T, I>;
    }

    template<class T, class I = std::size_t>
    inline constexpr bool is_contiguous_container_v =
        std::conjunction<internal::hasSize<T>, internal::isContiguousIndexable<T, I>>::value;

    // ************************************** select_last ************************************** //

    template<typename... Ts>
    struct select_last
    {
        // FIXME: Replace `std::enable_if<true, T>` by `std::type_identity<T>` when moving to C++20
        using type = typename decltype((std::enable_if<true, Ts>{}, ...))::type;
    };

    template<typename... Ts>
    using select_last_t = typename select_last<Ts...>::type;

    // ************************************* remove_cvref ************************************** //

    // TODO: Remove it when moving to C++20 as it has been added to the standard library
    template<class T>
    struct remove_cvref
    {
        using type = std::remove_cv_t<std::remove_reference_t<T>>;
    };

    template<typename T>
    using remove_cvref_t = typename remove_cvref<T>::type;

    // ********************************* is_base_of_template_v ********************************* //

    namespace internal
    {
        /// @sa For reference, see:
        /// https://stackoverflow.com/a/34672753/4820605
        template<template<typename...> class Base, typename Derived>
        struct IsBaseOfTemplateImpl
        {
            template<typename... Ts>
            static constexpr std::true_type test(const Base<Ts...> *);
            static constexpr std::false_type test(...);
            using value = decltype(test(std::declval<Derived *>()));
        };
    }

    template<template<typename...> class Base, typename Derived>
    using is_base_of_template_v = typename internal::IsBaseOfTemplateImpl<Base, Derived>::value;

    // *************************************** is_vector *************************************** //

    template<typename T>
    struct is_vector : std::false_type
    {
    };

    template<typename T, typename A>
    struct is_vector<std::vector<T, A>> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_vector_v = is_vector<std::decay_t<T>>::value;

    // **************************************** is_array *************************************** //

    template<typename T>
    struct is_array : std::false_type
    {
    };

    template<typename T, int I>
    struct is_array<std::array<T, I>> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_array_v = is_array<std::decay_t<T>>::value;

    // **************************************** is_map ***************************************** //

    namespace internal::is_map
    {
        template<typename K, typename T, typename C, typename A>
        std::true_type test(const std::map<K, T, C, A> *);
        template<typename K, typename T, typename H, typename C, typename A>
        std::true_type test(const std::unordered_map<K, T, H, C, A> *);
        std::false_type test(...);

        template<typename T>
        struct Test : public decltype(test(std::declval<std::add_pointer_t<T>>()))
        {
        };
    }

    template<typename T, typename Enable = void>
    struct is_map : std::false_type
    {
    };

    template<typename T>
    struct is_map<T, typename std::enable_if_t<internal::is_map::Test<T>::value>> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_map_v = is_map<T>::value;

    // ************************************ is_eigen_vector ************************************ //

    namespace internal::is_eigen_vector
    {
        template<typename T, int RowsAtCompileTime>
        std::true_type test(const Eigen::Matrix<T, RowsAtCompileTime, 1> *);
        template<typename T, int RowsAtCompileTime>
        std::true_type test(const Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1>> *);
        template<typename T, int RowsAtCompileTime>
        std::true_type test(const Eigen::Ref<const Eigen::Matrix<T, RowsAtCompileTime, 1>> *);
        template<typename T,
                 int RowsAtCompileTime,
                 int ColsAtCompileTime,
                 int BlockRowsAtCompileTime>
        std::true_type
        test(const Eigen::VectorBlock<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>,
                                      BlockRowsAtCompileTime> *);
        template<typename T,
                 int RowsAtCompileTime,
                 int ColsAtCompileTime,
                 int BlockRowsAtCompileTime>
        std::true_type
        test(const Eigen::VectorBlock<const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>,
                                      BlockRowsAtCompileTime> *);
        std::false_type test(...);

        template<typename T>
        struct Test : public decltype(test(std::declval<std::add_pointer_t<T>>()))
        {
        };
    }

    template<typename T, typename Enable = void>
    struct is_eigen_vector : std::false_type
    {
    };

    template<typename T>
    struct is_eigen_vector<T,
                           typename std::enable_if_t<internal::is_eigen_vector::Test<T>::value>> :
    std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_eigen_vector_v = is_eigen_vector<T>::value;

    // ************************************* is_eigen_ref ************************************** //

    namespace internal::is_eigen_ref
    {
        template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type
        test(const Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>> *);
        template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type
        test(const Eigen::Ref<const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>> *);
        template<typename T,
                 int RowsAtCompileTime,
                 int ColsAtCompileTime,
                 int BlockRowsAtCompileTime>
        std::true_type
        test(const Eigen::VectorBlock<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>,
                                      BlockRowsAtCompileTime> *);
        template<typename T,
                 int RowsAtCompileTime,
                 int ColsAtCompileTime,
                 int BlockRowsAtCompileTime>
        std::true_type
        test(const Eigen::VectorBlock<const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>,
                                      BlockRowsAtCompileTime> *);
        template<typename T,
                 int RowsAtCompileTime,
                 int ColsAtCompileTime,
                 int BlockRowsAtCompileTime,
                 int BlockColsAtCompileTime>
        std::true_type
        test(const Eigen::Block<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>,
                                BlockRowsAtCompileTime,
                                BlockColsAtCompileTime> *);
        template<typename T,
                 int RowsAtCompileTime,
                 int ColsAtCompileTime,
                 int BlockRowsAtCompileTime,
                 int BlockColsAtCompileTime>
        std::true_type
        test(const Eigen::Block<const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>,
                                BlockRowsAtCompileTime,
                                BlockColsAtCompileTime> *);
        std::false_type test(...);

        template<typename T>
        struct Test : public decltype(test(std::declval<std::add_pointer_t<T>>()))
        {
        };
    }

    template<typename T, typename Enable = void>
    struct is_eigen_ref : std::false_type
    {
    };

    template<typename T>
    struct is_eigen_ref<T, typename std::enable_if_t<internal::is_eigen_ref::Test<T>::value>> :
    std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_eigen_ref_v = is_eigen_ref<std::decay_t<T>>::value;

    // *************************************** is_eigen **************************************** //

    namespace internal::is_eigen_plain
    {
        template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> *);
        template<typename T>
        std::enable_if_t<is_eigen_ref_v<T>, std::true_type> test(const T *);
        std::false_type test(...);

        template<typename T>
        struct Test : public decltype(test(std::declval<std::add_pointer_t<T>>()))
        {
        };
    }

    template<typename T, typename Enable = void>
    struct is_eigen_plain : public std::false_type
    {
    };

    template<typename T>
    struct is_eigen_plain<T, typename std::enable_if_t<internal::is_eigen_plain::Test<T>::value>> :
    std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_eigen_plain_v = is_eigen_plain<std::decay_t<T>>::value;

    template<class T>
    inline constexpr bool is_eigen_object_v =
        std::disjunction<is_eigen_plain<T>, is_eigen_ref<T>>::value;

    template<typename Derived>
    inline constexpr bool is_eigen_any_v =
        std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Derived>>, std::decay_t<Derived>>;

    // ********************************** is_pinocchio_joint_ ********************************** //

#define IS_PINOCCHIO_JOINT_ENABLE_IF(type, name)                                          \
    IS_PINOCCHIO_JOINT_DETAILS(type, name)                                                \
                                                                                          \
    namespace internal::is_pinocchio_joint_##name                                         \
    {                                                                                     \
        template<typename T>                                                              \
        struct Test : public decltype(test(std::declval<std::add_pointer_t<T>>()))        \
        {                                                                                 \
        };                                                                                \
    }                                                                                     \
                                                                                          \
    template<typename T, typename Enable = void>                                          \
    struct is_pinocchio_joint_##name : public std::false_type                             \
    {                                                                                     \
    };                                                                                    \
                                                                                          \
    template<typename T>                                                                  \
    struct is_pinocchio_joint_##name<                                                     \
        T,                                                                                \
        typename std::enable_if_t<internal::is_pinocchio_joint_##name::Test<T>::value>> : \
    std::true_type                                                                        \
    {                                                                                     \
    };                                                                                    \
                                                                                          \
    template<typename T>                                                                  \
    inline constexpr bool is_pinocchio_joint_##name##_v = is_pinocchio_joint_##name<T>::value;

#define IS_PINOCCHIO_JOINT_DETAILS(type, name)                                          \
    namespace internal::is_pinocchio_joint_##name                                       \
    {                                                                                   \
        template<typename Scalar, int Options>                                          \
        std::true_type test(const pinocchio::JointModel##type##Tpl<Scalar, Options> *); \
        template<typename Scalar, int Options>                                          \
        std::true_type test(const pinocchio::JointData##type##Tpl<Scalar, Options> *);  \
        std::false_type test(...);                                                      \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(FreeFlyer, freeflyer)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Spherical, spherical)
    IS_PINOCCHIO_JOINT_ENABLE_IF(SphericalZYX, spherical_zyx)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Translation, translation)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Planar, planar)
    IS_PINOCCHIO_JOINT_ENABLE_IF(PrismaticUnaligned, prismatic_unaligned)
    IS_PINOCCHIO_JOINT_ENABLE_IF(RevoluteUnaligned, revolute_unaligned)
    IS_PINOCCHIO_JOINT_ENABLE_IF(RevoluteUnboundedUnaligned, revolute_unbounded_unaligned)

#undef IS_PINOCCHIO_JOINT_DETAILS
#define IS_PINOCCHIO_JOINT_DETAILS(type, name)                                                \
    namespace internal::is_pinocchio_joint_##name                                             \
    {                                                                                         \
        template<typename Scalar, int Options, int axis>                                      \
        std::true_type test(const pinocchio::JointModel##type##Tpl<Scalar, Options, axis> *); \
        template<typename Scalar, int Options, int axis>                                      \
        std::true_type test(const pinocchio::JointData##type##Tpl<Scalar, Options, axis> *);  \
        std::false_type test(...);                                                            \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(Prismatic, prismatic)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Revolute, revolute)
    IS_PINOCCHIO_JOINT_ENABLE_IF(RevoluteUnbounded, revolute_unbounded)

#undef IS_PINOCCHIO_JOINT_DETAILS
#define IS_PINOCCHIO_JOINT_DETAILS(type, name)                       \
    namespace internal::is_pinocchio_joint_##name                    \
    {                                                                \
        template<typename T>                                         \
        std::true_type test(const pinocchio::JointModel##type<T> *); \
        template<typename T>                                         \
        std::true_type test(const pinocchio::JointData##type<T> *);  \
        std::false_type test(...);                                   \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(Mimic, mimic)

#undef IS_PINOCCHIO_JOINT_DETAILS
#define IS_PINOCCHIO_JOINT_DETAILS(type, name)                                              \
    namespace internal::is_pinocchio_joint_##name                                           \
    {                                                                                       \
        template<typename Scalar,                                                           \
                 int Options,                                                               \
                 template<typename S, int O>                                                \
                 class JointCollectionTpl>                                                  \
        std::true_type test(                                                                \
            const pinocchio::JointModel##type##Tpl<Scalar, Options, JointCollectionTpl> *); \
        template<typename Scalar,                                                           \
                 int Options,                                                               \
                 template<typename S, int O>                                                \
                 class JointCollectionTpl>                                                  \
        std::true_type test(                                                                \
            const pinocchio::JointData##type##Tpl<Scalar, Options, JointCollectionTpl> *);  \
        std::false_type test(...);                                                          \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(Composite, composite)

#undef IS_PINOCCHIO_JOINT_DETAILS
#undef IS_PINOCCHIO_JOINT_ENABLE_IF
}

#endif  // JIMINY_TRAITS_H
