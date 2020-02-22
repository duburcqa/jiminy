# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_python VERSION ${BUILD_VERSION})

# Python install helper
function(deploy_python_package TARGET_NAME)
    install(CODE "EXECUTE_PROCESS (COMMAND pip install --upgrade --user .
                                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME})")
endfunction()

# Compile and install python bindings
add_subdirectory(${LIBRARY_NAME}_pywrap)

# Install native python packages
deploy_python_package(${LIBRARY_NAME}_py)