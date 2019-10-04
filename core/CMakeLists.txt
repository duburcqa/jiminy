# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_core VERSION ${BUILD_VERSION})

# Find library and headers
find_package(Boost REQUIRED)
find_package(urdfdom REQUIRED)

# Pinocchio specific stuff
SET(COMPILE_FLAGS "-D PINOCCHIO_WITH_URDFDOM")
add_definitions(${COMPILE_FLAGS})

# Add sources
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AbstractIODevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/MemoryDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TelemetryData.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TelemetrySender.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TelemetryRecorder.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Utilities.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AbstractSensor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AbstractController.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ControllerFunctor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Sensor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Model.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Engine.cc"
)

# Make library
add_library(${PROJECT_NAME} STATIC ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${BUILD_VERSION}
)

# Set include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
)

# Link with other libraries
target_link_libraries(${PROJECT_NAME} PUBLIC urdfdom_model boost_system)

# Activate c++14
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)


# Install
install(DIRECTORY "include/${LIBRARY_NAME}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS ${PROJECT_NAME}
        EXPORT  ${PROJECT_NAME}Config
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
export(TARGETS ${PROJECT_NAME}
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
)
install(EXPORT
        ${PROJECT_NAME}Config
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${BUILD_VERSION}
    COMPATIBILITY ExactVersion
)
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
)