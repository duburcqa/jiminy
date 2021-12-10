# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_pywrap VERSION ${BUILD_VERSION})

# Compile and install python bindings
add_subdirectory(${LIBRARY_NAME}_pywrap)

# Copy Python bindings as Python native 'core' submodule
install(CODE "file(GLOB_RECURSE src_file_list FOLLOW_SYMLINKS
                   LIST_DIRECTORIES false
                   \"${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_pywrap/*core*\")
              list(FILTER src_file_list INCLUDE REGEX \".*\.(so|dll|pyd|dylib).*\")
              set(src_file_symlinks)
              set(build_pypi_core_dir \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core\")
              foreach(src_file \${src_file_list})
                  get_filename_component(src_file_real \"\${src_file}\" REALPATH)
                  get_filename_component(src_file_name \"\${src_file}\" NAME)
                  if(IS_SYMLINK \"\${src_file}\")
                      get_filename_component(src_file_name_orig \"\${src_file_real}\" NAME)
                      list(APPEND src_file_symlinks \"\${build_pypi_core_dir}/\${src_file_name_orig}\")
                  endif()
                  file(COPY \"\${src_file_real}/\"
                       DESTINATION \"\${build_pypi_core_dir}/\${src_file_name}\")
              endforeach()
              if (src_file_symlinks)
                  file(REMOVE \${src_file_symlinks})
              endif()"
)

# Bundle headers, core library, and cmake config into Python package `core` submodule.
# Disable forbidden access to target LOCATION property is necessary, since expression
# generator in `install(CODE ...)` commands is only support from Cmake 3.14 and upward,
# which is newer than default version available on Ubuntu 18.04. For reference, see:
# https://stackoverflow.com/a/56528615/4820605
get_target_property(CORE_SOURCE_DIR ${LIBRARY_NAME} SOURCE_DIR)
get_target_property(PYWRAP_SOURCE_DIR ${LIBRARY_NAME}_pywrap SOURCE_DIR)
get_property(CORE_BINARY_DIR TARGET ${LIBRARY_NAME} PROPERTY BINARY_DIR)
if(WIN32)
     # Using delayed variable extension on Windows to resolve build type at install time, otherwise it would be undefined
     set(CORE_BINARY_DIR "${CORE_BINARY_DIR}/\${CMAKE_INSTALL_CONFIG_NAME}")
endif()
if(WIN32)
     set(CORE_LIB "${CORE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
     set(CORE_OBJ "${CORE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY_NAME}.lib")
else()
     get_property(CORE_SOVERSION TARGET ${LIBRARY_NAME} PROPERTY SOVERSION)
     if(APPLE)
          set(CORE_LIB "${CORE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY_NAME}.${CORE_SOVERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
     else()
          set(CORE_LIB "${CORE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}.${CORE_SOVERSION}")
     endif()
endif()
install(CODE "foreach(target_source \"${CORE_SOURCE_DIR}\" \"${PYWRAP_SOURCE_DIR}\")
                  file(GLOB_RECURSE archive_files_list FOLLOW_SYMLINKS
                       LIST_DIRECTORIES false
                       RELATIVE \"\${target_source}/include\"
                       \"\${target_source}/include/*\"
                  )
                  foreach(archive_file \${archive_files_list})
                      get_filename_component(archive_file_real \"\${archive_file}\" REALPATH
                                             BASE_DIR \"\${target_source}/include\")
                      file(COPY \"\${archive_file_real}/\"
                           DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/include/\${archive_file}\")
                  endforeach()
              endforeach()
              get_filename_component(CORE_LIB_REAL \"${CORE_LIB}\" REALPATH)
              get_filename_component(CORE_LIB_NAME \"${CORE_LIB}\" NAME)
              file(COPY \"\${CORE_LIB_REAL}/\"
                   DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/lib/\${CORE_LIB_NAME}\")
              if(WIN32)
                  get_filename_component(CORE_OBJ_REAL \"${CORE_OBJ}\" REALPATH)
                  get_filename_component(CORE_OBJ_NAME \"${CORE_OBJ}\" NAME)
                  file(COPY \"\${CORE_OBJ_REAL}/\"
                       DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/lib/\${CORE_OBJ_NAME}\")
              endif()
              file(COPY \"${CMAKE_SOURCE_DIR}/build_tools/cmake/jiminyConfig.cmake\"
                   DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/cmake\")
              include(CMakePackageConfigHelpers)
              write_basic_package_version_file(
                  ${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/cmake/${LIBRARY_NAME}ConfigVersion.cmake
                  VERSION ${BUILD_VERSION}
                  COMPATIBILITY ${COMPATIBILITY_VERSION}
              )"
)

################ jiminy_py ################

# Build Python wheels and archives
buildPythonWheel("python/${LIBRARY_NAME}_py"
                 "${CMAKE_BINARY_DIR}/pypi/dist/${LIBRARY_NAME}_py")

# Install Python package jiminy_py
deployPythonPackage("${LIBRARY_NAME}_py")

################ gym_jiminy ################

buildPythonWheel("python/gym_${LIBRARY_NAME}/common"
                 "python/gym_${LIBRARY_NAME}/toolbox"
                 "python/gym_${LIBRARY_NAME}/envs"
                 "python/gym_${LIBRARY_NAME}/rllib"
                 "${CMAKE_BINARY_DIR}/pypi/dist/gym_${LIBRARY_NAME}")
deployPythonPackageDevelop("python/gym_${LIBRARY_NAME}/common"
                           "python/gym_${LIBRARY_NAME}/toolbox"
                           "python/gym_${LIBRARY_NAME}/envs"
                           "python/gym_${LIBRARY_NAME}/rllib")
