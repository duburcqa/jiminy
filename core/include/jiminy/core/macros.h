
#ifndef JIMINY_MACRO_H
#define JIMINY_MACRO_H

#include <stddef.h>   // `size_t`
#include <algorithm>  // `std::copy`, `std::search`, `std::find`
#include <iostream>   // `std::cerr`, `std::endl`
#include <iterator>   // `std::reverse_iterator`

#include <boost/current_function.hpp>  // `BOOST_CURRENT_FUNCTION`


// ************************************* Generic utilities ************************************* //

#define DISABLE_COPY(className)                  \
    className(const className & other) = delete; \
    className & operator=(const className & other) = delete;

// ******************************** Symbol visibility utilities ******************************** //

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

// ******************************* Error and warnings utilities ******************************** //

namespace jiminy::internal
{
    template<size_t FL, size_t PFL>
    const char * extractFunctionName(const char (&func)[FL], const char (&pretty_func)[PFL])
    {
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

#define PRINT_ERROR(...)                                                                 \
    std::cerr << "In " FILE_LINE ": In "                                                 \
              << jiminy::internal::extractFunctionName(__func__, BOOST_CURRENT_FUNCTION) \
              << ":\n\x1b[1;31merror:\x1b[0m " << toString(__VA_ARGS__) << std::endl

#ifdef NDEBUG
#    define PRINT_WARNING(...)
#else
#    define PRINT_WARNING(...)                                                               \
        std::cerr << "In " FILE_LINE ": In "                                                 \
                  << jiminy::internal::extractFunctionName(__func__, BOOST_CURRENT_FUNCTION) \
                  << ":\n\x1b[1;93mwarning:\x1b[0m " << toString(__VA_ARGS__) << std::endl
#endif

#endif  // JIMINY_MACRO_H
