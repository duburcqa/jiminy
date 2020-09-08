# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_core VERSION ${BUILD_VERSION})

# Find libraries and headers
find_package(Boost REQUIRED COMPONENTS system filesystem date_time thread)
find_package(urdfdom NO_MODULE) # It is impossible to specify the version is not exported in cmake config...
if(urdfdom_FOUND)
    # Using pkgconfig is the only way to get the library version...
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(_URDFDOM QUIET "urdfdom")
        if ((_URDFDOM_FOUND) AND (NOT _URDFDOM_PREFIX STREQUAL "/usr"))
            set(urdfdom_VERSION "${_URDFDOM_VERSION}")
        endif()
    endif()
endif()
if((NOT urdfdom_FOUND) AND (WIN32))
    message(FATAL_ERROR "urdfdom not found, CMake will exit.")
elseif(urdfdom_FOUND)
    message("-- Found urdfdom: version ${urdfdom_VERSION}")
endif((NOT urdfdom_FOUND) AND (WIN32))
find_package(Eigen3 3.3.4 NO_MODULE)  # It adds target include lib Eigen3::Eigen
if((NOT Eigen3_FOUND) AND (WIN32))
    message(FATAL_ERROR "Eigen3 not found, CMake will exit.")
elseif(Eigen3_FOUND)
    message("-- Found Eigen3: version ${Eigen3_VERSION}")
endif((NOT Eigen3_FOUND) AND (WIN32))
find_package(pinocchio 2.4.0 NO_MODULE)  # Pinocchio was header-only before 2.4.0. It adds target include lib pinocchio::pinocchio.
if(pinocchio_FOUND)
    message("-- Found pinocchio: version ${pinocchio_VERSION}")
elseif(pinocchio_FOUND)
    find_package(hpp-fcl 1.4.0 REQUIRED NO_MODULE)  # Old pinocchio releases do not export hpp-fcl dependency
endif(pinocchio_FOUND)

# Pinocchio-specific stuffs
set(COMPILE_FLAGS "-DPINOCCHIO_WITH_URDFDOM -DPINOCCHIO_WITH_HPP_FCL")
if (urdfdom_VERSION) # Assume version strictly older than 0.4.0 if urdfdom version undefined
    if ("${urdfdom_VERSION}" VERSION_GREATER_EQUAL "0.4.0")
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR")
    endif()
    if ("${urdfdom_VERSION}" VERSION_GREATER "0.4.2")
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -DPINOCCHIO_URDFDOM_USE_STD_SHARED_PTR")
    endif()
endif()
add_definitions(${COMPILE_FLAGS})

# Add sources
set(SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Constants.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Utilities.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/AbstractIODevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/MemoryDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/FileDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/JsonLoader.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/JsonWriter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetryData.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetrySender.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetryRecorder.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/Model.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/AbstractConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/AbstractMotor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/BasicMotors.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/AbstractSensor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/BasicSensors.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/FixedFrameConstraint.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/robot/Robot.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/control/AbstractController.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/EngineMultiRobot.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/Engine.cc"
)

# Create the library
add_library(${PROJECT_NAME} STATIC ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${BUILD_VERSION}
)

# Set include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
)

# Configure jsoncpp dependency.
# Note that defining only BUILD_INTERFACE will prevent the header files to be exported,
# thus not making them available by a side-project depending on Jiminy. However, since
# jsoncpp is installed on the system as an external project, those side-projects must
# simply call find_package(jsoncpp) and it will be included automatically.
add_dependencies(${PROJECT_NAME} jsoncpp_external)
EXTERNALPROJECT_GET_PROPERTY(jsoncpp_external SOURCE_DIR)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
    $<BUILD_INTERFACE:${SOURCE_DIR}/include>
)

# Include headers from other libraries (in such a way to avoid any warnings compiling them)
list(REMOVE_ITEM Boost_LIBRARIES ${BOOST_PYTHON_LIB})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "${Boost_INCLUDE_DIRS}")
if(urdfdom_FOUND)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "${urdfdom_INCLUDE_DIRS}")
endif()

# Linking with other libraries (in such a way to avoid any warnings compiling them)
if(pinocchio_FOUND)
    target_link_libraries_system(${PROJECT_NAME} pinocchio::pinocchio)
else(pinocchio_FOUND)
    target_link_libraries_system(${PROJECT_NAME} hpp-fcl::hpp-fcl)
endif(pinocchio_FOUND)
if(urdfdom_FOUND)
    if (NOT "${urdfdom_LIBRARIES}" MATCHES ".*tinyxml.*")
        list(APPEND urdfdom_LIBRARIES "tinyxml")
    endif()
    target_link_libraries(${PROJECT_NAME} "${urdfdom_LIBRARIES}")
else(urdfdom_FOUND)
    target_link_libraries(${PROJECT_NAME} urdfdom_model) # Fallback to raw library name
endif(urdfdom_FOUND)
if(Eigen3_FOUND)
    target_link_libraries_system(${PROJECT_NAME} Eigen3::Eigen)
endif()
target_link_libraries(${PROJECT_NAME} jsoncpp_lib "${Boost_LIBRARIES}")

# Activate C++14 on Linux (not properly supported by MSVC 2015)
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
