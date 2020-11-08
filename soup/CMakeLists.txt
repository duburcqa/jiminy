# Import external project utilities
include(ExternalProject)

# Set the compilation flags
if(NOT WIN32)
    set(CMAKE_COMPILE_FLAGS_EXTERNAL "-w -fPIC")
else(NOT WIN32)
    set(CMAKE_COMPILE_FLAGS_EXTERNAL "/EHsc /bigobj -W1 -fPIC")
endif(NOT WIN32)

# Set fake credential
set(GIT_CREDENTIAL_EXTERNAL user.name=cmake;user.email=external.project@cmake.com)

# Propagate the global build type
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (isMultiConfig)
     set(EXTERNALPROJECT_BUILD_TYPE_CMD --config $<CONFIG>)
else()
     set(EXTERNALPROJECT_BUILD_TYPE_CMD -DCMAKE_BUILD_TYPE=$<CONFIG>)
endif()

# Add all SOUP
file(GLOB subdirs_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(subdir ${subdirs_list})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        add_subdirectory(${subdir})
    endif()
endforeach()
