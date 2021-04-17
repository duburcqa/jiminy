
#ifndef JIMINY_MACRO_H
#define JIMINY_MACRO_H

#include <type_traits>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "pinocchio/fwd.hpp"                          // To avoid having to include it everywhere
#include "pinocchio/multibody/joint/fwd.hpp"          // `pinocchio::JointModel ## type ## Tpl`, `pinocchio::JointData ## type ## Tpl`

#include <Eigen/Core>

#include <boost/current_function.hpp>

// `pinocchio::JointModelMimic`, `pinocchio::JointDataMimic`
namespace pinocchio
{
    /* Note that multiple forward declaration is not an error, so no big deal
       if future pinocchio versions start to forward declare mimic joints. */
    template<class JointModel> struct JointModelMimic;
    template<class JointData> struct JointDataMimic;
}


namespace jiminy
{
    // **************** Generic template utilities ******************

    // https://stackoverflow.com/a/34672753/4820605
    template<template<typename...> class base, typename derived>
    struct is_base_of_template_impl
    {
        template<typename... Ts>
        static constexpr std::true_type test(base<Ts...> const *);
        static constexpr std::false_type test(...);
        using type = decltype(test(std::declval<derived *>()));
    };

    template<template <typename...> class base, typename derived>
    using is_base_of_template = typename is_base_of_template_impl<base, derived>::type;

    // https://stackoverflow.com/a/37227316/4820605
    template <class F, class... Args>
    void do_for(F f, Args... args)
    {
        (f(args), ...);
    }

    template<class F, class dF=std::decay_t<F> >
    auto not_F(F && f)
    {
        return [f=std::forward<F>(f)](auto && ... args) mutable
               ->decltype(!std::declval<std::result_of_t<dF &(decltype(args)...)> >())  // Optional, adds SFINAE
               {
                   return !f(decltype(args)(args)...);
               };
    }

    // ================= enable_shared_from_this ====================

    template<typename Base>
    inline std::shared_ptr<Base>
    shared_from_base(std::enable_shared_from_this<Base> * base)
    {
        return base->shared_from_this();
    }

    template<typename Base>
    inline std::shared_ptr<Base const>
    shared_from_base(std::enable_shared_from_this<Base> const * base)
    {
        return base->shared_from_this();
    }

    template<typename T>
    inline std::shared_ptr<T>
    shared_from(T * derived)
    {
        return std::static_pointer_cast<T>(shared_from_base(derived));
    }

    // ======================== is_vector ===========================

    template<typename T>
    struct is_vector : std::false_type {};

    template<typename T>
    struct is_vector<std::vector<T> > : std::true_type {};

    template<typename T>
    constexpr bool is_vector_v = is_vector<T>::value;  // `inline` variables are not supported by gcc<7.3

    // ========================== is_map ============================

    namespace isMapDetail
    {
        template<typename K, typename T>
        std::true_type test(std::map<K, T> const *);
        template<typename K, typename T>
        std::true_type test(std::unordered_map<K, T> const *);
        std::false_type test(...);
    }

    template<typename T>
    struct isMap : public decltype(isMapDetail::test(std::declval<T *>())) {};

    template<typename T, typename Enable = void>
    struct is_map : std::false_type {};

    template<typename T>
    struct is_map<T, typename std::enable_if<isMap<T>::value>::type> : std::true_type {};

    template<typename T>
    constexpr bool is_map_v = is_map<T>::value;

    // ========================= is_eigen ===========================

    namespace isEigenObjectDetail
    {
        template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> const *);
        template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> > const *);
        template<typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> const> const *);
        std::false_type test(...);
    }

    template<typename T>
    struct isEigenObject : public decltype(isEigenObjectDetail::test(std::declval<T *>())) {};

    template<typename T, typename Enable = void>
    struct is_eigen : public std::false_type {};

    template<typename T>
    struct is_eigen<T, typename std::enable_if_t<isEigenObject<T>::value> > : std::true_type {};

    template<typename T>
    constexpr bool is_eigen_v = is_eigen<T>::value;

    // ====================== is_not_eigen_expr =======================

    // Check it is eigen object has its own storage, otherwise it is an expression
    // https://stackoverflow.com/questions/53770832/type-trait-to-check-if-an-eigen-type-is-an-expression-without-storage-or-a-mat
    template<typename T>
    struct is_not_eigen_expr
    : std::is_base_of<Eigen::PlainObjectBase<std::decay_t<T> >, std::decay_t<T> >
    {};

    // ====================== is_eigen_vector =======================

    namespace isEigenVectorDetail
    {
        template<typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Matrix<T, RowsAtCompileTime, 1> const *);
        template<typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> > const *);
        template<typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> const> const *);
        std::false_type test(...);
    }

    template<typename T>
    struct isEigenVector : public decltype(isEigenVectorDetail::test(std::declval<T *>())) {};

    template<typename T, typename Enable = void>
    struct is_eigen_vector : std::false_type {};

    template<typename T>
    struct is_eigen_vector<T, typename std::enable_if_t<isEigenVector<T>::value> > : std::true_type {};

    template<typename T>
    constexpr bool is_eigen_vector_v = is_eigen_vector<T>::value;

    // =================== is_pinocchio_joint_* ===================

    #define IS_PINOCCHIO_JOINT_ENABLE_IF(type, name) \
    IS_PINOCCHIO_JOINT_DETAIL(type, name) \
     \
    template<typename T> \
    struct isPinocchioJoint ## type : \
        public decltype(isPinocchioJoint ## type ## Detail ::test(std::declval<T *>())) {}; \
     \
    template<typename T, typename Enable = void> \
    struct is_pinocchio_joint_ ## name : public std::false_type {}; \
     \
    template<typename T> \
    struct is_pinocchio_joint_ ## name <T, typename std::enable_if_t<isPinocchioJoint ## type <T>::value> > : std::true_type {}; \
     \
    template<typename T> \
    constexpr bool is_pinocchio_joint_ ## name ## _v = is_pinocchio_joint_ ## name <T>::value;

    #define IS_PINOCCHIO_JOINT_DETAIL(type, name) \
    namespace isPinocchioJoint ## type ## Detail \
    { \
        template<typename Scalar, int Options> \
        std::true_type test(pinocchio::JointModel ## type ## Tpl<Scalar, Options> const *); \
        template<typename Scalar, int Options> \
        std::true_type test(pinocchio::JointData ## type ## Tpl<Scalar, Options> const *); \
        std::false_type test(...); \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(FreeFlyer, freeflyer)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Spherical, spherical)
    IS_PINOCCHIO_JOINT_ENABLE_IF(SphericalZYX, spherical_zyx)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Translation, translation)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Planar, planar)
    IS_PINOCCHIO_JOINT_ENABLE_IF(PrismaticUnaligned, prismatic_unaligned)
    IS_PINOCCHIO_JOINT_ENABLE_IF(RevoluteUnaligned, revolute_unaligned)
    IS_PINOCCHIO_JOINT_ENABLE_IF(RevoluteUnboundedUnaligned, revolute_unbounded_unaligned)

    #undef IS_PINOCCHIO_JOINT_DETAIL
    #define IS_PINOCCHIO_JOINT_DETAIL(type, name) \
    namespace isPinocchioJoint ## type ## Detail \
    { \
        template<typename Scalar, int Options, int axis> \
        std::true_type test(pinocchio::JointModel ## type ## Tpl<Scalar, Options, axis> const *); \
        template<typename Scalar, int Options, int axis> \
        std::true_type test(pinocchio::JointData ## type ## Tpl<Scalar, Options, axis> const *); \
        std::false_type test(...); \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(Prismatic, prismatic)
    IS_PINOCCHIO_JOINT_ENABLE_IF(Revolute, revolute)
    IS_PINOCCHIO_JOINT_ENABLE_IF(RevoluteUnbounded, revolute_unbounded)

    #undef IS_PINOCCHIO_JOINT_DETAIL
    #define IS_PINOCCHIO_JOINT_DETAIL(type, name) \
    namespace isPinocchioJoint ## type ## Detail \
    { \
        template<typename T> \
        std::true_type test(pinocchio::JointModel ## type<T> const *); \
        template<typename T> \
        std::true_type test(pinocchio::JointData ## type<T> const *); \
        std::false_type test(...); \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(Mimic, mimic)

    #undef IS_PINOCCHIO_JOINT_DETAIL
    #define IS_PINOCCHIO_JOINT_DETAIL(type, name) \
    namespace isPinocchioJoint ## type ## Detail \
    { \
        template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl> \
        std::true_type test(pinocchio::JointModel ## type ## Tpl<Scalar, Options, JointCollectionTpl> const *); \
        template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl> \
        std::true_type test(pinocchio::JointData ## type ## Tpl<Scalar, Options, JointCollectionTpl> const *); \
        std::false_type test(...); \
    }

    IS_PINOCCHIO_JOINT_ENABLE_IF(Composite, composite)

    #undef IS_PINOCCHIO_JOINT_DETAIL
    #undef IS_PINOCCHIO_JOINT_ENABLE_IF

    // ************* Error message generation ****************

    template<typename... Args>
    std::string to_string(Args &&... args)
    {
        std::ostringstream sstr;
        using List = int[];
        (void)List{0, ( (void)(sstr << args), 0 ) ... };
        return sstr.str();
    }

    #define STRINGIFY_DETAIL(x) #x
    #define STRINGIFY(x) STRINGIFY_DETAIL(x)

    #define FILE_LINE __FILE__ ":" STRINGIFY(__LINE__)

    /* ANSI escape codes is used here as a cross-platform way to color text.
       For reference, see:
       https://solarianprogrammer.com/2019/04/08/c-programming-ansi-escape-codes-windows-macos-linux-terminals/ */

    #define PRINT_ERROR(...) \
    std::cerr << "In " FILE_LINE ": In " << BOOST_CURRENT_FUNCTION << ":\n\x1b[1;31merror:\x1b[0m " << to_string(__VA_ARGS__) << std::endl

    #ifdef NDEBUG
        #define PRINT_WARNING(...)
    #else
        #define PRINT_WARNING(...) \
        std::cerr << "In " FILE_LINE ": In " << BOOST_CURRENT_FUNCTION << ":\n\x1b[1;93mwarning:\x1b[0m " << to_string(__VA_ARGS__) << std::endl
    #endif
}

#endif  // JIMINY_MACRO_H
