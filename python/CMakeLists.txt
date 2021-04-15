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
              list(FILTER src_file_list INCLUDE REGEX \".*\.(so|dll|pyd)$\")
              foreach(src_file \${src_file_list})
                  get_filename_component(src_file_real \"\${src_file}\" REALPATH)
                  get_filename_component(src_file_name \"\${src_file}\" NAME)
                  file(COPY \"\${src_file_real}/\"
                       DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/\${src_file_name}\")
              endforeach()"
)

# Bundle headers and core library into Python package `core` submodule.
# Disable forbidden access to target LOCATION property is necessary, since expression
# generator in `install(CODE ...)` commands is only support from Cmake 3.14 and upward,
# which is newer than default version available on Ubuntu 18.04. For reference, see:
# https://stackoverflow.com/a/56528615/4820605
get_target_property(CORE_SOURCE_DIR ${LIBRARY_NAME} SOURCE_DIR)
get_property(CORE_BINARY_DIR TARGET ${LIBRARY_NAME} PROPERTY BINARY_DIR)
get_property(CORE_SOVERSION TARGET ${LIBRARY_NAME} PROPERTY SOVERSION)
set(CORE_LIB "${CORE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}.${CORE_SOVERSION}")
install(CODE "set(CORE_LIB ${CORE_LIB})
              file(GLOB_RECURSE archive_files_list FOLLOW_SYMLINKS
                   LIST_DIRECTORIES false
                   RELATIVE \"${CORE_SOURCE_DIR}/include\"
                   \"${CORE_SOURCE_DIR}/include/*\"
              )
              foreach(archive_file \${archive_files_list})
                  get_filename_component(archive_file_real \"\${archive_file}\" REALPATH
                                         BASE_DIR \"${CORE_SOURCE_DIR}/include\")
                  file(COPY \"\${archive_file_real}/\"
                       DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/include/\${archive_file}\")
              endforeach()
              get_filename_component(CORE_LIB_REAL \"\${CORE_LIB}\" REALPATH)
              get_filename_component(CORE_LIB_NAME \"\${CORE_LIB}\" NAME)
              file(COPY \"\${CORE_LIB_REAL}/\"
                   DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/lib/\${CORE_LIB_NAME}\")"
     )

################ jiminy_py ################

# Build Python wheels and archives
buildPythonWheel("python/${LIBRARY_NAME}_py"
                 "${CMAKE_BINARY_DIR}/pypi/dist/${LIBRARY_NAME}_py")

# Install Python package jiminy_py
deployPythonPackage("${LIBRARY_NAME}_py")

################ gym_jiminy ################

buildPythonWheel("python/gym_${LIBRARY_NAME}/common"
                 "python/gym_${LIBRARY_NAME}/envs"
                 "python/gym_${LIBRARY_NAME}/toolbox"
                 "${CMAKE_BINARY_DIR}/pypi/dist/gym_${LIBRARY_NAME}")
deployPythonPackageDevelop("python/gym_${LIBRARY_NAME}/common"
                           "python/gym_${LIBRARY_NAME}/envs"
                           "python/gym_${LIBRARY_NAME}/toolbox")
