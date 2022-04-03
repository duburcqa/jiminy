# Minimum version required
cmake_minimum_required(VERSION 3.10)

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
find_package(pinocchio 2.5.6 REQUIRED NO_MODULE NO_CMAKE_SYSTEM_PATH)  # Pinocchio v2.5.6 fixes 'aba' overwritting 'data.a_gf'
find_package(hpp-fcl 1.7.1 REQUIRED NO_MODULE NO_CMAKE_SYSTEM_PATH)  # hpp-fcl >= 1.7.1 adds collision geometry pointers
find_package(Eigen3 3.3.0 REQUIRED NO_MODULE)  # It adds the target Eigen3::Eigen

# Pinocchio-specific stuffs
set(COMPILE_FLAGS "-DPINOCCHIO_WITH_URDFDOM -DPINOCCHIO_WITH_HPP_FCL")
if (urdfdom_VERSION) # Assume version strictly older than 0.4.0 if urdfdom version undefined
    if (urdfdom_VERSION VERSION_GREATER_EQUAL "0.4.0")
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR")
    endif()
    if (urdfdom_VERSION VERSION_GREATER "0.4.2")
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -DPINOCCHIO_URDFDOM_USE_STD_SHARED_PTR")
    endif()
endif()
add_definitions("${COMPILE_FLAGS}")

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

# Create the library
add_library(${PROJECT_NAME}-object OBJECT ${SRC})
add_library(${PROJECT_NAME} STATIC $<TARGET_OBJECTS:${PROJECT_NAME}-object>)
add_library(${LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${PROJECT_NAME}-object>)

# Activate C++17 for this target
set_target_properties(${PROJECT_NAME}-object PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_compile_features(${PROJECT_NAME}-object PUBLIC cxx_std_17)

# Set include directory
target_include_directories(${PROJECT_NAME}-object PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
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

# Include headers from other libraries (in such a way to avoid any warnings compiling them)
target_include_directories(${PROJECT_NAME}-object SYSTEM PUBLIC "${Boost_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME}-object SYSTEM PUBLIC "${urdfdom_INCLUDE_DIRS}")
get_target_property(Eigen3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${PROJECT_NAME}-object SYSTEM PUBLIC ${Eigen3_INCLUDE_DIRS})

# `target_link_libraries` for object libraries is not available for Cmake < 3.12,
# so instead the dependencies must be added for both the static and the shared library.
foreach(target "${PROJECT_NAME}" "${LIBRARY_NAME}")
    # Set library version
    set_target_properties(${target} PROPERTIES
        VERSION ${BUILD_VERSION}
        SOVERSION ${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}
    )

    # Propagate include directories from object manually is necessary for Cmake < 3.12
    target_include_directories(${target}
        SYSTEM PUBLIC
        $<TARGET_PROPERTY:${PROJECT_NAME}-object,INTERFACE_INCLUDE_DIRECTORIES>)

    # Re-activate C++17 because it does not propagate properly for Cmake < 3.12
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_compile_features(${target} PUBLIC cxx_std_17)

    # Linking with other libraries (in such a way to avoid any warnings compiling them)
    target_link_libraries_system(${target} pinocchio::pinocchio)
    target_link_libraries_system(${target} hpp-fcl::hpp-fcl)
    if (NOT "${urdfdom_LIBRARIES}" MATCHES ".*tinyxml.*")
        list(APPEND urdfdom_LIBRARIES "tinyxml")
    endif()
    target_link_libraries(${target} "${urdfdom_LIBRARIES}")
    target_link_libraries(${target} Eigen3::Eigen)
    target_link_libraries(${target} jsoncpp::jsoncpp hdf5::hdf5_cpp hdf5::hdf5 hdf5::zlib)  # Beware the order is critical !
    target_link_libraries(${target} ${Boost_LIBRARIES})
    # Link some libraries that are not automatically linked with HDF5 and assimp (through hppfcl) respectively
    if(UNIX AND NOT APPLE)
        target_link_libraries(${target} ${CMAKE_DL_LIBS} -lrt)
    endif()
endforeach()

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
