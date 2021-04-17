# Enable link rpath to find shared library dependencies at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Get Python executable
if(DEFINED PYTHON_EXECUTABLE)
    get_filename_component(_PYTHON_PATH "${PYTHON_EXECUTABLE}" DIRECTORY)
    get_filename_component(_PYTHON_NAME "${PYTHON_EXECUTABLE}" NAME)
    find_program(Python_EXECUTABLE "${_PYTHON_NAME}" PATHS "${_PYTHON_PATH}" NO_DEFAULT_PATH)
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
    if (jiminy_FIND_REQUIRED)
        message(FATAL_ERROR "Python executable not found, CMake will exit.")
    else()
        return()
    endif()
endif()

# Make sure jiminy Python module is available
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import importlib; print(int(importlib.util.find_spec('jiminy_py') is not None), end='')"
                OUTPUT_VARIABLE jiminy_FOUND)
if (NOT jiminy_FOUND)
    if (jiminy_FIND_REQUIRED)
        message(FATAL_ERROR "`jiminy_py` Python module not found, CMake will exit.")
    else()
        return()
    endif()
endif()

# Find jiminy library and headers.
# Note that jiminy is compiled under C++17 using either old or new CXX11 ABI.
# Make sure very project dependencies are compiled for the same CXX11 ABI
# otherwise segfaults may occur. It should be fine for the standard library,
# but not for precompiled boost libraries such as boost::filesystem.
# https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.__version__, end='')"
                OUTPUT_VARIABLE jiminy_VERSION)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.get_include(), end='')"
                OUTPUT_VARIABLE jiminy_INCLUDE_DIRS)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "import jiminy_py; print(jiminy_py.get_libraries(), end='')"
                OUTPUT_VARIABLE jiminy_LIBRARIES)

# Define compilation options and definitions
set(jiminy_DEFINITIONS EIGENPY_STATIC URDFDOM_STATIC HPP_FCL_STATIC PINOCCHIO_STATIC)
if(WIN32)
    set(jiminy_DEFINITIONS ${jiminy_DEFINITIONS} _USE_MATH_DEFINES=1 NOMINMAX)
    set(jiminy_OPTIONS /EHsc /bigobj /Zc:__cplusplus /permissive- /wd4996 /wd4554)
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
