# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_pywrap VERSION ${BUILD_VERSION})

# Add cpp sources
set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/Module.cc")

# Python sources et setup
file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/python/*.py")

# Make library
add_library(${PROJECT_NAME} SHARED ${SRC})
add_definitions(-DBOOST_PYTHON_STATIC_LIB)
add_definitions(-DBOOST_PYTHON_STATIC_MODULE)
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${BUILD_VERSION}
)

# Substitute Docstring @copydoc flags with C++ Doxygen documentations
pythonDocstingSubstitution()

# Set include directory (build folder because of the substitution)
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/${PROJECT_NAME}/include/>"
)

# Link with other libraries
target_link_libraries(${PROJECT_NAME}
        PUBLIC ${LIBRARY_NAME}_core eigenpy "${BOOST_PYTHON_LIB}"
)
target_include_directories(${PROJECT_NAME}
        SYSTEM PRIVATE ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
)

# Activate c++14
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Copy Python bindings submodule
install(TARGETS ${PROJECT_NAME}
        DESTINATION "${CMAKE_BINARY_DIR}/pypi/jiminy_py/src/jiminy_py/core"
)
install(FILES ${${PROJECT_NAME}_PYTHON_SOURCES}
        DESTINATION "${CMAKE_BINARY_DIR}/pypi/jiminy_py/src/jiminy_py/core"
)

# Install C++ headers
install(DIRECTORY "include/${LIBRARY_NAME}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Generate Cmake configuration files used by find_package
install(TARGETS ${PROJECT_NAME}
        EXPORT  ${PROJECT_NAME}Config
        DESTINATION "${PYTHON_SITELIB}/${LIBRARY_NAME}"
)

exportCmakeConfigFiles()
