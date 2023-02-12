# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Project name
project(${LIBRARY_NAME}_core VERSION ${BUILD_VERSION})

# Find dependencies
find_package(Boost REQUIRED COMPONENTS system filesystem serialization date_time thread)
find_package(urdfdom REQUIRED NO_MODULE NO_CMAKE_SYSTEM_PATH)  # It is impossible to specify the version because it is not exported in cmake config files...
find_package(PkgConfig QUIET)  # Using pkgconfig is the only way to get the library version...
if (PkgConfig_FOUND)
    pkg_check_modules(_URDFDOM QUIET "urdfdom")
    if ((_URDFDOM_FOUND) AND (NOT _URDFDOM_PREFIX STREQUAL "/usr"))
        set(urdfdom_VERSION "${_URDFDOM_VERSION}")
    endif()
endif()
if(urdfdom_VERSION)
    message(STATUS "Found urdfdom: version ${urdfdom_VERSION}")
else()
    message(STATUS "Found urdfdom")
endif()
find_package(pinocchio 2.6.13 REQUIRED NO_MODULE NO_CMAKE_SYSTEM_PATH)
find_package(hpp-fcl 2.0.0 REQUIRED NO_MODULE NO_CMAKE_SYSTEM_PATH)
find_package(Eigen3 3.3.0 REQUIRED NO_MODULE)

# Specialize jiminy core configuration file
set(JIMINY_CONFIG_IN ${CMAKE_SOURCE_DIR}/build_tools/cmake/jiminyConfig.cmake.in)
set(JIMINY_CONFIG_OUT ${CMAKE_BINARY_DIR}/cmake/jiminyConfig.cmake)
configure_file(${JIMINY_CONFIG_IN} ${JIMINY_CONFIG_OUT} @ONLY)

# Enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FULL}")

# Add sources
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Constants.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utilities/Helpers.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utilities/Pinocchio.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utilities/Json.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utilities/Random.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/AbstractIODevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/MemoryDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/FileDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/JsonLoader.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/JsonWriter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/Serialization.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetryData.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetrySender.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetryRecorder.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/constraints/AbstractConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/constraints/JointConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/constraints/FixedFrameConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/constraints/DistanceConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/constraints/SphereConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/constraints/WheelConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/Model.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/AbstractMotor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/BasicMotors.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/AbstractSensor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/BasicSensors.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/Robot.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/control/AbstractController.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver/ConstraintSolvers.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stepper/AbstractStepper.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stepper/EulerExplicitStepper.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stepper/AbstractRungeKuttaStepper.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stepper/RungeKutta4Stepper.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/stepper/RungeKuttaDOPRIStepper.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/System.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/EngineMultiRobot.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/Engine.cc"
)

# Export all symbols when building shared library to enable building extension module
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Create intermediary object
add_library(${PROJECT_NAME}-object OBJECT ${SRC})

# Activate C++17 for this target
set_target_properties(${PROJECT_NAME}-object PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_compile_features(${PROJECT_NAME}-object PUBLIC cxx_std_17)

# Pinocchio-specific stuffs.
# Note that the maximum cpp version of Eigen must be consistent with
# jiminy, otherwise memory alignment issues may occur.
set(COMPILE_FLAGS EIGEN_MAX_CPP_VER=11 PINOCCHIO_WITH_URDFDOM PINOCCHIO_WITH_HPP_FCL)
if (urdfdom_VERSION) # Assume version strictly older than 0.4.0 if urdfdom version undefined
    if (urdfdom_VERSION VERSION_GREATER_EQUAL "0.4.0")
        list(APPEND COMPILE_FLAGS PINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR)
    endif()
    if (urdfdom_VERSION VERSION_GREATER "0.4.2")
        list(APPEND COMPILE_FLAGS PINOCCHIO_URDFDOM_USE_STD_SHARED_PTR)
    endif()
endif()
target_compile_definitions(${PROJECT_NAME}-object PUBLIC ${COMPILE_FLAGS})

# Boost-specific stuffs. It must be defined before pinocchio does it,
# otherwise it messes with LTO.
target_compile_definitions(${PROJECT_NAME}-object PUBLIC
    BOOST_MPL_LIMIT_VECTOR_SIZE=30
)

# Set include directory
target_include_directories(${PROJECT_NAME}-object PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)

# Configure jsoncpp and hdf5 dependencies.
# Note that defining only BUILD_INTERFACE will prevent the header files to be exported,
# thus not making them available by a side-project depending on Jiminy. However, since
# jsoncpp is installed on the system as an external project, those side-projects must
# simply call find_package(jsoncpp) and it will be included automatically.
add_dependencies(${PROJECT_NAME}-object jsoncpp_external hdf5_external)
externalproject_get_property(jsoncpp_external INSTALL_DIR)
target_include_directories(${PROJECT_NAME}-object SYSTEM PUBLIC
    $<BUILD_INTERFACE:${INSTALL_DIR}/include>
)

# Linking with other libraries (in such a way to avoid any warnings compiling them)
target_link_libraries_system(${PROJECT_NAME}-object pinocchio::pinocchio hpp-fcl::hpp-fcl Eigen3::Eigen)
if (NOT "${urdfdom_LIBRARIES}" MATCHES ".*tinyxml.*")
    list(APPEND urdfdom_LIBRARIES "tinyxml")
endif()
target_link_libraries(${PROJECT_NAME}-object ${urdfdom_LIBRARIES})
target_link_libraries(${PROJECT_NAME}-object jsoncpp::jsoncpp hdf5::hdf5_cpp hdf5::hdf5 hdf5::zlib)  # Beware the order is critical !
target_link_libraries(${PROJECT_NAME}-object ${Boost_LIBRARIES})
# Link some libraries that are not automatically linked with HDF5 and assimp (through hppfcl) respectively
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME}-object ${CMAKE_DL_LIBS} -lrt)
endif()

# Create both static and shared libraries from intermediary object
add_library(${PROJECT_NAME} STATIC $<TARGET_OBJECTS:${PROJECT_NAME}-object>)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}-object)
add_library(${LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${PROJECT_NAME}-object>)
target_link_libraries(${LIBRARY_NAME} ${PROJECT_NAME}-object)

# Set library version
foreach(target "${PROJECT_NAME}" "${LIBRARY_NAME}")
    set_target_properties(${target} PROPERTIES
        VERSION ${BUILD_VERSION}
        SOVERSION ${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}
    )
endforeach()

# Build C++ unit tests
option(BUILD_TESTING "Build the C++ unit tests." ON)
if(BUILD_TESTING)
    include(CTest)
    include(GoogleTest)
    add_subdirectory(unit)
endif()

# Build C++ examples
option(BUILD_EXAMPLES "Build the C++ examples." ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install C++ headers
install(DIRECTORY "include/${LIBRARY_NAME}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Generate Cmake configuration files used by find_package
install(TARGETS ${PROJECT_NAME}-object ${PROJECT_NAME}
        EXPORT  ${PROJECT_NAME}Config
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        OBJECTS DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

exportCmakeConfigFiles(${PROJECT_NAME}-object ${PROJECT_NAME})
