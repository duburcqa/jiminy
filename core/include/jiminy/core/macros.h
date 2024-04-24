
#ifndef JIMINY_MACRO_H
#define JIMINY_MACRO_H

#include <stddef.h>   // `size_t`
#include <algorithm>  // `std::copy`, `std::search`, `std::find`
#include <iostream>   // `std::cerr`, `std::endl`
#include <iterator>   // `std::reverse_iterator`

#include <boost/current_function.hpp>  // `BOOST_CURRENT_FUNCTION`


// ************************************* Generic utilities ************************************* //

#define JIMINY_DISABLE_COPY(className)           \
    className(const className & other) = delete; \
    className & operator=(const className & other) = delete;

// ******************************** Symbol visibility utilities ******************************** //

#if defined _WIN32 || defined __CYGWIN__
// On Microsoft Windows, use dllimport and dllexport to tag symbols
#    define JIMINY_DLLIMPORT __declspec(dllimport)
#    define JIMINY_DLLEXPORT __declspec(dllexport)
#    define JIMINY_TEMPLATE_DLLIMPORT
#    define JIMINY_TEMPLATE_DLLEXPORT
#    define JIMINY_STATIC_MEMBER_DLLIMPORT
#    define JIMINY_STATIC_MEMBER_DLLEXPORT JIMINY_DLLEXPORT
#    define JIMINY_TEMPLATE_INSTANTIATION_DLLIMPORT JIMINY_DLLIMPORT
#    define JIMINY_TEMPLATE_INSTANTIATION_DLLEXPORT JIMINY_DLLEXPORT
#else
// On Linux, tag symbols using de-facto standard visibility attribute extension
#    define JIMINY_DLLIMPORT __attribute__((visibility("default")))
#    define JIMINY_DLLEXPORT __attribute__((visibility("default")))
#    define JIMINY_TEMPLATE_DLLIMPORT JIMINY_DLLIMPORT
#    define JIMINY_TEMPLATE_DLLEXPORT JIMINY_DLLEXPORT
#    define JIMINY_STATIC_MEMBER_DLLIMPORT
#    define JIMINY_STATIC_MEMBER_DLLEXPORT
#    define JIMINY_TEMPLATE_INSTANTIATION_DLLIMPORT
#    define JIMINY_TEMPLATE_INSTANTIATION_DLLEXPORT
#endif

// Define DLLAPI to import or export depending on whether one is building or using the library
#ifdef EXPORT_SYMBOLS
#    define JIMINY_DLLAPI JIMINY_DLLEXPORT
#    define JIMINY_TEMPLATE_DLLAPI JIMINY_TEMPLATE_DLLEXPORT
#    define JIMINY_STATIC_MEMBER_DLLAPI JIMINY_STATIC_MEMBER_DLLEXPORT
#    define JIMINY_TEMPLATE_INSTANTIATION_DLLAPI JIMINY_TEMPLATE_INSTANTIATION_DLLEXPORT
#else
#    define JIMINY_DLLAPI JIMINY_DLLIMPORT
#    define JIMINY_TEMPLATE_DLLAPI JIMINY_TEMPLATE_DLLIMPORT
#    define JIMINY_STATIC_MEMBER_DLLAPI JIMINY_STATIC_MEMBER_DLLIMPORT
#    define JIMINY_TEMPLATE_INSTANTIATION_DLLAPI JIMINY_TEMPLATE_INSTANTIATION_DLLIMPORT
#endif

// ******************************* Error and warnings utilities ******************************** //

namespace jiminy::internal
{
    template<size_t FL, size_t PFL>
    const char * extractFunctionName(const char (&func)[FL], const char (&pretty_func)[PFL])
    {
        // FIXME: Make the whole method 'constexpr' when moving to C++20
        using reverse_ptr = std::reverse_iterator<const char *>;
        thread_local static char result[PFL];
        const char * locFuncName =
            std::search(pretty_func, pretty_func + PFL - 1, func, func + FL - 1);
        const char * locClassName =
            std::find(reverse_ptr(locFuncName), reverse_ptr(pretty_func), ' ').base();
        const char * endFuncName = std::find(locFuncName, pretty_func + PFL - 1, '(');
        std::copy(locClassName, endFuncName, result);
        return result;
    }
}

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x) STRINGIFY_DETAIL(x)

#define FILE_LINE __FILE__ ":" STRINGIFY(__LINE__)

/* ANSI escape codes is used here as a cross-platform way to color text. For reference, see:
   https://solarianprogrammer.com/2019/04/08/c-programming-ansi-escape-codes-windows-macos-linux-terminals/
*/
#define JIMINY_THROW(exception, ...)                                                      \
    throw exception(                                                                      \
        toString(jiminy::internal::extractFunctionName(__func__, BOOST_CURRENT_FUNCTION), \
                 "(" FILE_LINE "):\n",                                                    \
                 __VA_ARGS__))

#define JIMINY_WARNING(...)                                                              \
    std::cout << "\x1b[1;93mWARNING\x1b[0m:"                                             \
              << jiminy::internal::extractFunctionName(__func__, BOOST_CURRENT_FUNCTION) \
              << "(" FILE_LINE "):\n"                                                    \
              << toString(__VA_ARGS__) << std::endl

#endif  // JIMINY_MACRO_H
