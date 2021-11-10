# Import external project utilities
include(ExternalProject)

# Set the compilation flags.
# Use the same flags than jiminy itself, but disable all warnings.
set(CMAKE_COMPILE_FLAGS_EXTERNAL "${CMAKE_CXX_FLAGS_RELEASE} -w")

# Set fake credential
set(GIT_CREDENTIAL_EXTERNAL user.name=cmake;user.email=external.project@cmake.com)

# Propagate the global build type.
# Note that it is already done automatically for multi-config generators,
# so it is only necessary for the single-config ones.
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT isMultiConfig)
     set(EXTERNALPROJECT_BUILD_TYPE_CMD -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

# Propagate OSX configuration
if(APPLE)
    string(REPLACE ";" "$<SEMICOLON>" CMAKE_OSX_ARCHITECTURES_ "${CMAKE_OSX_ARCHITECTURES}")
    set(EXTERNALPROJECT_OSX_CONFIG
        -DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES_}
        -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()

# Add all SOUP
file(GLOB subdirs_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(subdir ${subdirs_list})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        add_subdirectory(${subdir})
    endif()
endforeach()
