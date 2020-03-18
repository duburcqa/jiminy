include(${CMAKE_SOURCE_DIR}/build_tools/cmake/externalProjectUtils.cmake)

# Set the compilation flags
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS_EXTERNAL "-w")
else(NOT WIN32)
    set(CMAKE_CXX_FLAGS_EXTERNAL "/EHsc /bigobj -W1")
endif(NOT WIN32)

# Set fake credential
set(GIT_CREDENTIAL_EXTERNAL user.name=cmake;user.email=external.project@cmake.com)

# Add all SOUP
SUBDIRLIST(${CMAKE_CURRENT_SOURCE_DIR} subdirs_list)
foreach(subdir ${subdirs_list})
    add_subdirectory(${subdir})
endforeach()
