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
                       DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/\${src_file_name}\")
              endforeach()"
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
