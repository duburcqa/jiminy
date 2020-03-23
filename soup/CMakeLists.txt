include(${CMAKE_SOURCE_DIR}/build_tools/cmake/externalProjectUtils.cmake)

# Set the compilation flags
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS_EXTERNAL "-w")
else(NOT WIN32)
    set(CMAKE_CXX_FLAGS_EXTERNAL "/EHsc /bigobj -W1")
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
SUBDIRLIST(${CMAKE_CURRENT_SOURCE_DIR} subdirs_list)
foreach(subdir ${subdirs_list})
    add_subdirectory(${subdir})
endforeach()
