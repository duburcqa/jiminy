
#ifndef JIMINY_MACRO_H
#define JIMINY_MACRO_H

#include <type_traits>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>

#include "pinocchio/fwd.hpp"

#include <Eigen/Core>

#include <boost/current_function.hpp>


namespace jiminy
{
    // **************** Generic template utilities ******************

    // https://stackoverflow.com/a/34672753/4820605
    template < template <typename...> class base, typename derived>
    struct is_base_of_template_impl
    {
        template<typename... Ts>
        static constexpr std::true_type  test(base<Ts...> const *);
        static constexpr std::false_type test(...);
        using type = decltype(test(std::declval<derived *>()));
    };

    template < template <typename...> class base,typename derived>
    using is_base_of_template = typename is_base_of_template_impl<base,derived>::type;

    // https://stackoverflow.com/a/37227316/4820605
    template <class F, class... Args>
    void do_for(F f, Args... args) {
        int x[] = {(f(args), 0)...};
    }

    template<class F, class dF=std::decay_t<F> >
    auto not_F(F && f)
    {
        return [f=std::forward<F>(f)](auto && ... args) mutable
               ->decltype(!std::declval<std::result_of_t<dF &(decltype(args)...)> >()) // optional, adds sfinae
               {
                   return !f(decltype(args)(args)...);
               };
    }

    // ================= enable_shared_from_this ====================

    template <typename Base>
    inline std::shared_ptr<Base>
    shared_from_base(std::enable_shared_from_this<Base> * base)
    {
        return base->shared_from_this();
    }
    template <typename Base>
    inline std::shared_ptr<const Base>
    shared_from_base(std::enable_shared_from_this<Base> const * base)
    {
        return base->shared_from_this();
    }
    template <typename That>
    inline std::shared_ptr<That>
    shared_from(That * that)
    {
        return std::static_pointer_cast<That>(shared_from_base(that));
    }

    // ======================== is_vector ===========================

    template<typename T>
    struct is_vector : std::false_type {};

    template<typename T>
    struct is_vector<std::vector<T> > : std::true_type {};

    // ========================= is_eigen ===========================

    namespace isEigenObjectDetail {
        template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> const *);
        template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> > const *);
        template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> const> const *);
        std::false_type test(...);
    }

    template <typename T>
    struct isEigenObject :
        public decltype(isEigenObjectDetail::test(std::declval<T *>())) {};

    template<typename T, typename Enable = void>
    struct is_eigen : public std::false_type {};

    template<typename T>
    struct is_eigen<T, typename std::enable_if<isEigenObject<T>::value>::type> : std::true_type {};

    // ====================== is_not_eigen_expr =======================

    // Check it is eigen object has its own storage, otherwise it is an expression
    // https://stackoverflow.com/questions/53770832/type-trait-to-check-if-an-eigen-type-is-an-expression-without-storage-or-a-mat
    template<typename T>
    struct is_not_eigen_expr
    : std::is_base_of<Eigen::PlainObjectBase<std::decay_t<T> >, std::decay_t<T> >
    {};

    // ====================== is_eigen_vector =======================

    namespace isEigenVectorDetail {
        template <typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Matrix<T, RowsAtCompileTime, 1> const *);
        template <typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> > const *);
        template <typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> const> const *);
        std::false_type test(...);
    }

    template <typename T>
    struct isEigenVector : public decltype(isEigenVectorDetail::test(std::declval<T *>())) {};

    template<typename T, typename Enable = void>
    struct is_eigen_vector : std::false_type {};

    template<typename T>
    struct is_eigen_vector<T, typename std::enable_if<isEigenVector<T>::value>::type> : std::true_type {};

    // ========================== is_map ============================

    namespace isMapDetail {
        template<typename K, typename T>
        std::true_type test(std::map<K, T> const *);
        template<typename K, typename T>
        std::true_type test(std::unordered_map<K, T> const *);
        std::false_type test(...);
    }

    template <typename T>
    struct isMap : public decltype(isMapDetail::test(std::declval<T *>())) {};

    template<typename T, typename Enable = void>
    struct is_map : std::false_type {};

    template<typename T>
    struct is_map<T, typename std::enable_if<isMap<T>::value>::type> : std::true_type {};

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

    #define PRINT_ERROR(...) \
    std::cout << "In " FILE_LINE ": In " << BOOST_CURRENT_FUNCTION << ":\n\033[1;32merror:\033[0m " << to_string(__VA_ARGS__) << std::endl;

    #ifdef NDEBUG
    #define PRINT_WARNING(...)
    #else
    #define PRINT_WARNING(...) \
    std::cout << "In " FILE_LINE ": In " << BOOST_CURRENT_FUNCTION << ":\n\033[1;93mwarning:\033[0m " << to_string(__VA_ARGS__) << std::endl;
    #endif
}

#endif  // JIMINY_MACRO_H
