# Import external project utilities
include(ExternalProject)

# Set the compilation flags
if(NOT WIN32)
    set(CMAKE_COMPILE_FLAGS_EXTERNAL "-DNDEBUG -O3 -s -fPIC -w")
else(NOT WIN32)
    set(CMAKE_COMPILE_FLAGS_EXTERNAL "/EHsc /bigobj -DNDEBUG /O2")
endif(NOT WIN32)

# Set fake credential
set(GIT_CREDENTIAL_EXTERNAL user.name=cmake;user.email=external.project@cmake.com)

# Propagate the global build type.
# Note that it is already done automatically for multi-config generators,
# so it is only necessary for the single-config ones.
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT isMultiConfig)
     set(EXTERNALPROJECT_BUILD_TYPE_CMD -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

# Add all SOUP
file(GLOB subdirs_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(subdir ${subdirs_list})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        add_subdirectory(${subdir})
    endif()
endforeach()
