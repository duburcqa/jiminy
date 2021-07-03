# Enable link rpath to find shared library dependencies at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set Python find strategy to location by default
if(CMAKE_VERSION VERSION_GREATER "3.15")
    cmake_policy(SET CMP0094 NEW)
endif()

# Get Python executable
if(DEFINED PYTHON_EXECUTABLE)
    get_filename_component(_PYTHON_PATH "${PYTHON_EXECUTABLE}" DIRECTORY)
    get_filename_component(_PYTHON_NAME "${PYTHON_EXECUTABLE}" NAME)
    find_program(Python_EXECUTABLE "${_PYTHON_NAME}" PATHS "${_PYTHON_PATH}" NO_DEFAULT_PATH)
else()
    if(CMAKE_VERSION VERSION_LESS "3.12.4")
        find_program(Python_EXECUTABLE "python${PYTHON_REQUIRED_VERSION}")
    else()
        if(CMAKE_VERSION VERSION_LESS "3.14")
            set(Python_COMPONENTS_FIND Interpreter)
        else()
            set(Python_COMPONENTS_FIND Interpreter NumPy)
        endif()
        unset(Python_EXECUTABLE)
        unset(Python_EXECUTABLE CACHE)
        unset(_Python_EXECUTABLE)
        unset(_Python_EXECUTABLE CACHE)
        if(PYTHON_REQUIRED_VERSION)
            find_package(Python ${PYTHON_REQUIRED_VERSION} EXACT COMPONENTS ${Python_COMPONENTS_FIND})
        else()
            find_package(Python COMPONENTS ${Python_COMPONENTS_FIND})
        endif()
    endif()
endif()
if(NOT Python_EXECUTABLE)
    if (jiminy_FIND_REQUIRED)
        message(FATAL_ERROR "Python executable not found, CMake will exit.")
    else()
        set(jiminy_FOUND FALSE)
        return()
    endif()
endif()

# Make sure jiminy Python module is available
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import importlib; print(int(importlib.util.find_spec('jiminy_py') is not None), end='')"
                OUTPUT_VARIABLE jiminy_FOUND)
if(NOT jiminy_FOUND)
    if (jiminy_FIND_REQUIRED)
        message(FATAL_ERROR "`jiminy_py` Python module not found, CMake will exit.")
    else()
        return()
    endif()
endif()

# Define Python_NumPy_INCLUDE_DIRS if necessary
if (NOT Python_NumPy_INCLUDE_DIRS)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
        "import numpy; print(numpy.get_include(), end='')\n"
        OUTPUT_VARIABLE __numpy_path)
    find_path(Python_NumPy_INCLUDE_DIRS numpy/arrayobject.h
        HINTS "${__numpy_path}" "${Python_INCLUDE_DIRS}" NO_DEFAULT_PATH)
endif()

# Find jiminy library and headers.
# Note that jiminy is compiled under C++17 using either old or new CXX11 ABI.
# Make sure very project dependencies are compiled for the same CXX11 ABI
# otherwise segfaults may occur. It should be fine for the standard library,
# but not for precompiled boost libraries such as boost::filesystem.
# https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.get_include(), end='')"
                OUTPUT_VARIABLE jiminy_INCLUDE_DIRS)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.get_libraries(), end='')"
                OUTPUT_VARIABLE jiminy_LIBRARIES)

# Define compilation options and definitions
set(jiminy_DEFINITIONS
    EIGENPY_STATIC URDFDOM_STATIC HPP_FCL_STATIC PINOCCHIO_STATIC
    BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS=ON BOOST_MPL_LIMIT_VECTOR_SIZE=30 BOOST_MPL_LIMIT_LIST_SIZE=30
)
if(WIN32)
    set(jiminy_DEFINITIONS ${jiminy_DEFINITIONS} _USE_MATH_DEFINES=1 NOMINMAX)
    set(jiminy_OPTIONS /EHsc /bigobj /Zc:__cplusplus /permissive- /wd4996 /wd4554 /wd4005)
else()
    execute_process(COMMAND readelf --version-info "${jiminy_LIBRARIES}"
                    COMMAND grep -c "Name: CXXABI_1.3.9\\\|Name: GLIBCXX_3.4.21"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE CHECK_NEW_CXX11_ABI)
    set(jiminy_DEFINITIONS ${jiminy_DEFINITIONS} _GLIBCXX_USE_CXX11_ABI=$<BOOL:${CHECK_NEW_CXX11_ABI}>)
    set(jiminy_OPTIONS -fPIC)
endif()

# Define imported target
add_library(jiminy::core SHARED IMPORTED GLOBAL)
set_target_properties(jiminy::core PROPERTIES
    IMPORTED_LOCATION "${jiminy_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${jiminy_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS "${jiminy_DEFINITIONS}"
    INTERFACE_COMPILE_OPTIONS "${jiminy_OPTIONS}"
    INTERFACE_COMPILE_FEATURES cxx_std_17
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
if(WIN32)
    get_filename_component(jiminy_LIBDIR "${jiminy_LIBRARIES}" DIRECTORY)
    get_filename_component(jiminy_LIBNAME "${jiminy_LIBRARIES}" NAME_WE)
    set_target_properties(jiminy::core PROPERTIES
        IMPORTED_IMPLIB "${jiminy_LIBDIR}/${jiminy_LIBNAME}.lib"
    )
endif()

# Display import is success
message(STATUS "Found jiminy ('${jiminy_VERSION}'): ${jiminy_INCLUDE_DIRS} (${jiminy_LIBRARIES})")
