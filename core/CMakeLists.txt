# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_core VERSION ${BUILD_VERSION})

# Find libraries and headers
find_package(Boost REQUIRED COMPONENTS system)
find_package(urdfdom REQUIRED)
if(WIN32)
    find_package(Eigen3 3.3.4 REQUIRED NO_MODULE) # it adds target include lib Eigen3::Eigen
    message("-- Found Eigen3: version ${Eigen3_VERSION}")
endif()

# Pinocchio-specific stuffs
set(COMPILE_FLAGS "-DPINOCCHIO_WITH_URDFDOM")
if(WIN32)
    # Add flags specific to urdfdom >= 1.0.0 (Urdfdom does not support explicit version checking using cmake files so far...)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR -DPINOCCHIO_URDFDOM_USE_STD_SHARED_PTR")
endif()
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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AbstractMotor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AbstractController.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BasicSensors.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BasicMotors.cc"
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

# Link with other libraries (in such a way to avoid any warnings compiling them)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE "${Boost_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} urdfdom_model "${Boost_LIBRARIES}")
if(WIN32)
    target_link_libraries_system(${PROJECT_NAME} Eigen3::Eigen)
endif()

# Activate C++14
if(NOT WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
endif()

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
