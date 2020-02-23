# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Set the build version
set(BUILD_VERSION 1.0.1)

# Project and library name
project(jiminy VERSION ${BUILD_VERSION})
set(LIBRARY_NAME ${PROJECT_NAME})

# Set build environment and standard dependencies
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/base.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/doxygen.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/boostPythonDocstring.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/exportCmakeConfigFiles.cmake)

# Sub-projects
add_subdirectory(core)
add_subdirectory(python)
add_subdirectory(scenarios)
add_subdirectory(unit)

# Build documentation from Doxygen.
build_DOC()

# Build Python wheels and archives
function(buildPythonWheel TARGET_PATH)
    get_filename_component(TARGET_NAME ${TARGET_PATH} NAME_WE)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/${TARGET_PATH}
            DESTINATION "${CMAKE_BINARY_DIR}/pypi/"
            PATTERN "*.egg-info" EXCLUDE
    )
    install(FILES ${CMAKE_SOURCE_DIR}/README.md
            DESTINATION "${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME}"
            COMPONENT pypi
            EXCLUDE_FROM_ALL
    )
    install(CODE "file(REMOVE_RECURSE \"${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME}/dist\")
                  execute_process(COMMAND python setup.py sdist bdist_wheel
                                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME}
                  )
                  file(GLOB PYTHON_WHEEL_FILENAME \"${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME}/dist/*.whl\")
                  string(REPLACE \"-linux_\" \"-manylinux1_\" PYTHON_WHEEL_FILENAME_FIXED \"\${PYTHON_WHEEL_FILENAME}\")
                  file(RENAME \"\${PYTHON_WHEEL_FILENAME}\" \"\${PYTHON_WHEEL_FILENAME_FIXED}\")
                  "
            COMPONENT pypi
            EXCLUDE_FROM_ALL
    )
endfunction()

buildPythonWheel(python/${LIBRARY_NAME}_py)
buildPythonWheel(gym_${LIBRARY_NAME})

# Install native python packages
deployPythonPackage(${LIBRARY_NAME}_py)

# Install Python native packages in develop mode
deployPythonPackageDevelop(gym_${LIBRARY_NAME})

# Install the source cmake files
file(GLOB_RECURSE SOURCE_CMAKE_FILES "${CMAKE_SOURCE_DIR}/build_tools/cmake/Find*.cmake")
install(FILES ${SOURCE_CMAKE_FILES}
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
)
install(FILES ${CMAKE_SOURCE_DIR}/build_tools/cmake/base.cmake
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
        RENAME ${LIBRARY_NAME}_base.cmake
)
install(FILES ${CMAKE_SOURCE_DIR}/build_tools/cmake/boostPythonDocstring.cmake
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
)
