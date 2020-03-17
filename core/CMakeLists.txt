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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Utilities.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/AbstractIODevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/MemoryDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/FileDevice.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/JsonLoader.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/JsonWriter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetryData.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetrySender.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/telemetry/TelemetryRecorder.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/AbstractMotor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/BasicMotors.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/AbstractSensor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/BasicSensors.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/Model.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/control/AbstractController.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Engine.cc"
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

# Link with other libraries (in such a way to avoid any warnings compiling them)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE "${Boost_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} urdfdom_model "${Boost_LIBRARIES}")
if(WIN32)
    target_link_libraries_system(${PROJECT_NAME} Eigen3::Eigen)
endif()

# Configure jsoncpp dependency.
# Note that using BUILD_INTERFACE will prevent the source files to be exported,
# thus not making them accessible by a side-project depending on Jiminy.
# So here, we are assuming those side-projects do not need to include jsoncpp.
add_dependencies(${PROJECT_NAME} jsoncpp)
EXTERNALPROJECT_GET_PROPERTY(jsoncpp SOURCE_DIR)
EXTERNALPROJECT_GET_PROPERTY(jsoncpp BINARY_DIR)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
    $<BUILD_INTERFACE:${SOURCE_DIR}/include>
)
target_link_libraries(${PROJECT_NAME} "-L${BINARY_DIR}/lib -ljsoncpp")

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
