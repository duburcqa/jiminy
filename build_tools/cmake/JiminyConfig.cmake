# Enable link rpath to find shared library dependencies at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Get Python executable
if(DEFINED PYTHON_EXECUTABLE)
    find_program(Python_EXECUTABLE python PATHS "${PYTHON_EXECUTABLE}" NO_DEFAULT_PATH)
else()
    if(CMAKE_VERSION VERSION_LESS "3.12.4")
        find_program(Python_EXECUTABLE "python${PYTHON_REQUIRED_VERSION}")
    else()
        if(PYTHON_REQUIRED_VERSION)
            find_package(Python ${PYTHON_REQUIRED_VERSION} EXACT COMPONENTS Interpreter)
        else()
            find_package(Python COMPONENTS Interpreter)
        endif()
    endif()
endif()
if(NOT Python_EXECUTABLE)
    if (Python_FIND_REQUIRED)
        message(FATAL_ERROR "Python executable not found, CMake will exit.")
    else()
        return()
    endif()
endif()

# Make sure Jiminy Python module is available
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import importlib; print(int(importlib.util.find_spec('jiminy_py') is not None), end='')"
                OUTPUT_VARIABLE Jiminy_FOUND)
if (NOT Jiminy_FOUND)
    if (Python_FIND_REQUIRED)
        message(FATAL_ERROR "`jiminy_py` Python module not found, CMake will exit.")
    else()
        return()
    endif()
endif()

# Find Jiminy library and headers.
# Note that Jiminy is compiled under C++17 using either old or new CXX11 ABI.
# Make sure very project dependencies are compiled for the same CXX11 ABI
# otherwise segfaults may occur. It should be fine for the standard library,
# but not for precompiled boost libraries such as boost::filesystem.
# https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.__version__, end='')"
                OUTPUT_VARIABLE Jiminy_VERSION)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.get_include(), end='')"
                OUTPUT_VARIABLE Jiminy_INCLUDE_DIRS)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.get_libraries(), end='')"
                OUTPUT_VARIABLE Jiminy_LIBRARIES)
if(NOT WIN32)
    execute_process(COMMAND readelf --version-info "${Jiminy_LIBRARIES}"
                    COMMAND grep -c "Name: CXXABI_1.3.9\\\|Name: GLIBCXX_3.4.21"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE CHECK_NEW_CXX11_ABI)
    if(CHECK_NEW_CXX11_ABI EQUAL 2)
        set(Jiminy_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=1)
    else()
        set(Jiminy_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Define imported target
add_library(Jiminy SHARED IMPORTED GLOBAL)
set_target_properties(Jiminy PROPERTIES
    IMPORTED_LOCATION "${Jiminy_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${Jiminy_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS ${Jiminy_DEFINITIONS}
    INTERFACE_COMPILE_FEATURES cxx_std_17
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
