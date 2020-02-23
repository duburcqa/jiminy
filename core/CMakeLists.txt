# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_core VERSION ${BUILD_VERSION})

# Find library and headers
find_package(Boost REQUIRED)
find_package(urdfdom REQUIRED)
#find_package(Eigen3 3.2.8 REQUIRED NO_MODULE) # Not supported officially before 3.3.4

# Pinocchio specific stuff
SET(COMPILE_FLAGS "-D PINOCCHIO_WITH_URDFDOM")
add_definitions(${COMPILE_FLAGS})

# Add sources
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AbstractIODevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/MemoryDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/FileDevice.cc"
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
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
)

# Link with other libraries
target_link_libraries(${PROJECT_NAME} PUBLIC urdfdom_model boost_system) #Eigen3::Eigen)

# Activate c++14
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
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

exportCmakeConfigFiles()
