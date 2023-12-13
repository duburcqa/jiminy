#ifndef JIMINY_EXCEPTIONS_H
#define JIMINY_EXCEPTIONS_H

#include <algorithm>  // `std::copy`, `std::search`, `std::find`
#include <iostream>   // `std::cerr`, `std::endl`
#include <iterator>   // `std::reverse_iterator`
#include <sstream>    // `std::ostringstream`
#include <stdexcept>  // `std::runtime_error`, `std::logic_error`
#include <string>     // `std::string`

#include "jiminy/core/fwd.h"

#include <Eigen/Core>  // `Eigen::IOFormat`, `Eigen::FullPrecision`

#include "jiminy/core/traits.h"  // `is_eigen_v`

#include <boost/current_function.hpp>  // `BOOST_CURRENT_FUNCTION`


namespace jiminy
{
    // ********************************** Exception utilities ********************************** //

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

    // ********************************** Warnings utilities *********************************** //

    template<typename... Args>
    std::string toString(Args &&... args)
    {
        std::ostringstream sstr;
        auto format = [](const auto & var)
        {
            if constexpr (is_eigen_v<decltype(var)>)
            {
                static const Eigen::IOFormat k_heavy_fmt(
                    Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
                return var.format(k_heavy_fmt);
            }
            return var;
        };
        ((sstr << format(args)), ...);
        return sstr.str();
    }

    template<size_t FL, size_t PFL>
    const char * extractMethodName(const char (&function)[FL], const char (&prettyFunction)[PFL])
    {
        using reverse_ptr = std::reverse_iterator<const char *>;
        thread_local static char result[PFL];
        const char * locFuncName =
            std::search(prettyFunction, prettyFunction + PFL - 1, function, function + FL - 1);
        const char * locClassName =
            std::find(reverse_ptr(locFuncName), reverse_ptr(prettyFunction), ' ').base();
        const char * endFuncName = std::find(locFuncName, prettyFunction + PFL - 1, '(');
        std::copy(locClassName, endFuncName, result);
        return result;
    }

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x) STRINGIFY_DETAIL(x)

#define FILE_LINE __FILE__ ":" STRINGIFY(__LINE__)

    /* ANSI escape codes is used here as a cross-platform way to color text. For reference, see:
       https://solarianprogrammer.com/2019/04/08/c-programming-ansi-escape-codes-windows-macos-linux-terminals/
    */

#define PRINT_ERROR(...)                                                                        \
    std::cerr << "In " FILE_LINE ": In " << extractMethodName(__func__, BOOST_CURRENT_FUNCTION) \
              << ":\n\x1b[1;31merror:\x1b[0m " << toString(__VA_ARGS__) << std::endl

#ifdef NDEBUG
#    define PRINT_WARNING(...)
#else
#    define PRINT_WARNING(...)                                           \
        std::cerr << "In " FILE_LINE ": In "                             \
                  << extractMethodName(__func__, BOOST_CURRENT_FUNCTION) \
                  << ":\n\x1b[1;93mwarning:\x1b[0m " << toString(__VA_ARGS__) << std::endl
#endif
}  // namespace jiminy

#endif  // JIMINY_EXCEPTIONS_H
