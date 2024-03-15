# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Project name
project(${LIBRARY_NAME}_pywrap VERSION ${BUILD_VERSION})

# Find libraries and headers
find_package(eigenpy 2.9.0 REQUIRED NO_MODULE NO_CMAKE_SYSTEM_PATH)  # >=2.9.0 fully support vectorization

# Enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FULL}")

# Add cpp sources (in binary dir, since the sources are pre-processed to substitute docstring)
set(SRC
    "${CMAKE_CURRENT_BINARY_DIR}/src/compatibility.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/helpers.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/functors.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/generators.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/robot.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/sensors.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/motors.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/controllers.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/constraints.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/engine.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/module.cc"
)

# Make library
add_library(${PROJECT_NAME} SHARED ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    VERSION ${BUILD_VERSION}
    SUFFIX ${PYTHON_EXT_SUFFIX}
    OUTPUT_NAME ${PYTHON_LIBRARY_NAME}
)
target_compile_definitions(${PROJECT_NAME} PUBLIC
    PYTHON_LIBRARY_NAME=${PYTHON_LIBRARY_NAME}
)

# Substitute Docstring @copydoc flags with C++ Doxygen documentations
pythonDocstingSubstitution()

# Set include directory (build folder because of the substitution)
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
)

# Link with other libraries
target_include_directories(${PROJECT_NAME}
    SYSTEM PRIVATE ${Python_NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} ${LIBRARY_NAME}_core eigenpy::eigenpy "${BOOST_PYTHON_LIB}")

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
