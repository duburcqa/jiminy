function(buildPythonWheel TARGET_PATH)
    get_filename_component(TARGET_NAME ${TARGET_PATH} NAME_WE)
    get_filename_component(TARGET_DIR ${TARGET_PATH} DIRECTORY)
    install(CODE "file(GLOB_RECURSE src_file_list FOLLOW_SYMLINKS
                       LIST_DIRECTORIES false
                       RELATIVE \"${CMAKE_SOURCE_DIR}/${TARGET_DIR}\"
                       \"${CMAKE_SOURCE_DIR}/${TARGET_PATH}/*\"
                  )
                  list(FILTER src_file_list EXCLUDE REGEX \".*\.egg-info\")
                  list(FILTER src_file_list EXCLUDE REGEX \"unit\")
                  foreach(src_file \${src_file_list})
                      get_filename_component(src_file_real \"\${src_file}\" REALPATH
                                             BASE_DIR \"${CMAKE_SOURCE_DIR}/${TARGET_DIR}\")
                      file(COPY \"\${src_file_real}/\"
                           DESTINATION \"${CMAKE_BINARY_DIR}/pypi/\${src_file}\")
                  endforeach()"
           )

    # TODO: Use add_custom_command instead of install to enable auto-cleanup of copied files
#     add_custom_command(
#         OUTPUT  ${CMAKE_BINARY_DIR}/pypi
#         COMMAND ${CMAKE_COMMAND} -E copy_directory \"${CMAKE_SOURCE_DIR}/${TARGET_PATH}\" \"${CMAKE_BINARY_DIR}/pypi\"
#     )

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
    set_directory_properties(PROPERTIES "${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME}" ADDITIONAL_MAKE_CLEAN_FILES)
endfunction()