# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_pywrap VERSION ${BUILD_VERSION})

# Find libraries and headers
find_package(eigenpy 2.0.2 REQUIRED NO_MODULE NO_CMAKE_SYSTEM_PATH)

# Enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FULL}")

# Add cpp sources (in binary dir, since the sources are pre-processed to substitute docstring)
set(SRC
    "${CMAKE_CURRENT_BINARY_DIR}/src/Compatibility.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Helpers.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Functors.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Generators.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Robot.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Sensors.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Motors.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Controllers.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Constraints.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Engine.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/src/Module.cc")

# Make sure there is no undefined symbols, otherwise import will fail
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Make library
add_library(${PROJECT_NAME} SHARED ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    VERSION ${BUILD_VERSION}
    SUFFIX ${PYTHON_EXT_SUFFIX}
    OUTPUT_NAME ${PYTHON_LIBRARY_NAME}
)
target_compile_definitions(${PROJECT_NAME} PUBLIC
    "-DPYTHON_LIBRARY_NAME=${PYTHON_LIBRARY_NAME}"
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
