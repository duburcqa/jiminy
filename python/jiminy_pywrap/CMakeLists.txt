# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_pywrap VERSION ${BUILD_VERSION})

# Find libraries and headers
find_package(eigenpy 2.0.2 NO_MODULE) # It adds target include lib eigenpy::eigenpy
if((NOT eigenpy_FOUND) AND (WIN32))
    message(FATAL_ERROR "Eigenpy not found, CMake will exit.")
elseif(eigenpy_FOUND)
    message("-- Found eigenpy: version ${eigenpy_VERSION}")
endif((NOT eigenpy_FOUND) AND (WIN32))

# Add cpp sources
set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/Module.cc")

# Make library
add_library(${PROJECT_NAME} SHARED ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    VERSION ${BUILD_VERSION}
    SUFFIX ${PYTHON_EXT_SUFFIX}
    OUTPUT_NAME "${PROJECT_NAME}"
)

# Substitute Docstring @copydoc flags with C++ Doxygen documentations
pythonDocstingSubstitution()

# Set include directory (build folder because of the substitution)
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/${PROJECT_NAME}/include/>"
)

# Link with other libraries
target_include_directories(${PROJECT_NAME}
        SYSTEM PRIVATE ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} ${LIBRARY_NAME}_core "${BOOST_PYTHON_LIB}")
if(eigenpy_FOUND)
    target_link_libraries_system(${PROJECT_NAME} eigenpy::eigenpy)
else(eigenpy_FOUND)
    target_link_libraries(${PROJECT_NAME} eigenpy) # Fallback to raw library name
endif(eigenpy_FOUND)

# Activate C++14
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Install C++ headers
install(DIRECTORY "include/${LIBRARY_NAME}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Generate Cmake configuration files used by find_package
install(TARGETS ${PROJECT_NAME}
        EXPORT  ${PROJECT_NAME}Config
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

exportCmakeConfigFiles()
