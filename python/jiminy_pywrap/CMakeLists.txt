# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_pywrap VERSION ${BUILD_VERSION})

# Find libraries and headers
find_package(Boost QUIET REQUIRED)
if(${Boost_MINOR_VERSION} GREATER_EQUAL 67)
    find_package(Boost REQUIRED COMPONENTS
                 "python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}"
                 "numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
    set(BOOST_PYTHON_LIB "${Boost_LIBRARIES}")
else(${Boost_MINOR_VERSION} GREATER_EQUAL 67)
    if(${PYTHON_VERSION_MAJOR} EQUAL 3)
        set(BOOST_PYTHON_LIB "boost_numpy3;boost_python3")
    else(${PYTHON_VERSION_MAJOR} EQUAL 3)
        set(BOOST_PYTHON_LIB "boost_numpy;boost_python")
    endif(${PYTHON_VERSION_MAJOR} EQUAL 3)
endif(${Boost_MINOR_VERSION} GREATER_EQUAL 67)
message("-- Boost Python Libs: ${Boost_LIBRARIES}")

if(WIN32)
    find_package(eigenpy 2.0.3 REQUIRED NO_MODULE) # It adds target include lib eigenpy::eigenpy
    message("-- Found eigenpy: version ${eigenpy_VERSION}")
endif()

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
if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} eigenpy)
else(NOT WIN32)
    target_link_libraries_system(${PROJECT_NAME} eigenpy::eigenpy)
endif(NOT WIN32)

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
