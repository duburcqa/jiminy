function(buildPythonWheel)
    # The input arguments are [TARGET_PATH...], OUTPUT_DIR

    # Extract the output arguments
    # Cannot use ARGN directly with list() command, so copy it to a variable first.
    # https://stackoverflow.com/a/18534238/4820605
    set(ARGS ${ARGN})
    list(LENGTH ARGS NUM_ARGS)
    if(${NUM_ARGS} LESS 2)
        message(FATAL_ERROR "Please specify at least one TARGET_PATH and OUTPUT_DIR.")
    endif()
    list(GET ARGS -1 OUTPUT_DIR)
    list(REMOVE_AT ARGS -1)

    # Disable forbidden acces to target LOCATION property, since expression generator
    # in `install(CODE ...` is only support from Cmake 3.14 and upward, which is newer
    # than default version available on Ubuntu 18.04. For reference, see:
    # https://stackoverflow.com/a/56528615/4820605
    cmake_policy(SET CMP0026 OLD)

    # Generate wheel sequentially for each target
    foreach(TARGET_PATH IN LISTS ARGS)
        get_filename_component(TARGET_NAME ${TARGET_PATH} NAME_WE)
        get_filename_component(TARGET_DIR ${TARGET_PATH} DIRECTORY)
        get_target_property(CORE_SOURCE_DIR ${LIBRARY_NAME}_core SOURCE_DIR)
        get_property(CORE_LIBDIR TARGET ${LIBRARY_NAME}_core PROPERTY LOCATION)
        install(CODE "cmake_policy(SET CMP0053 NEW)
                      cmake_policy(SET CMP0011 NEW)
                      set(PROJECT_VERSION ${BUILD_VERSION})
                      set(PROJECT_INCLUDEDIR ${CORE_SOURCE_DIR}/include)
                      set(PROJECT_LIBDIR ${CORE_LIBDIR})
                      set(SOURCE_DIR ${CMAKE_SOURCE_DIR})
                      file(GLOB_RECURSE src_file_list FOLLOW_SYMLINKS
                          LIST_DIRECTORIES false
                          RELATIVE \"${CMAKE_SOURCE_DIR}/${TARGET_DIR}\"
                          \"${CMAKE_SOURCE_DIR}/${TARGET_PATH}/*\"
                      )
                      list(FILTER src_file_list EXCLUDE REGEX \".*\.egg-info\")
                      list(FILTER src_file_list EXCLUDE REGEX \"unit\")
                      list(FILTER src_file_list EXCLUDE REGEX \"__pycache__\")
                      list(FILTER src_file_list EXCLUDE REGEX \"mypy_cache\")
                      foreach(src_file \${src_file_list})
                          get_filename_component(src_file_real \"\${src_file}\" REALPATH
                                                 BASE_DIR \"${CMAKE_SOURCE_DIR}/${TARGET_DIR}\")
                          if(src_file_real MATCHES \".*\\.(txt|py|md|in|js|html|toml|json|urdf|xacro)\$\")
                              configure_file(\"\${src_file_real}\"
                                             \"${CMAKE_BINARY_DIR}/pypi/\${src_file}\" @ONLY)
                          else()
                              configure_file(\"\${src_file_real}\"
                                             \"${CMAKE_BINARY_DIR}/pypi/\${src_file}\" COPYONLY)
                          endif()
                      endforeach()"
            )

        # TODO: Use add_custom_command instead of install to enable auto-cleanup of copied files
        # add_custom_command(
        #     OUTPUT  ${CMAKE_BINARY_DIR}/pypi
        #     COMMAND ${CMAKE_COMMAND} -E copy_directory \"${CMAKE_SOURCE_DIR}/${TARGET_PATH}\" \"${CMAKE_BINARY_DIR}/pypi\"
        # )

        install(FILES ${CMAKE_SOURCE_DIR}/README.md
                DESTINATION "${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME}"
                COMPONENT pypi
                EXCLUDE_FROM_ALL
        )
        install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py clean --all
                                      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME})
                      execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
                                      --dist-dir \"${OUTPUT_DIR}\"
                                      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME})"
                COMPONENT pypi
                EXCLUDE_FROM_ALL
        )
        set_directory_properties(PROPERTIES "${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME}" ADDITIONAL_MAKE_CLEAN_FILES)
    endforeach()
endfunction()
