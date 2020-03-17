include(${CMAKE_SOURCE_DIR}/build_tools/cmake/externalProjectUtils.cmake)

# Add all SOUP
SUBDIRLIST(${CMAKE_CURRENT_SOURCE_DIR} subdirs_list)
foreach(subdir ${subdirs_list})
    add_subdirectory(${subdir})
endforeach()
