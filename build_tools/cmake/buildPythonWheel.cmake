function(buildPythonWheel TARGET_PATH)

    get_filename_component(TARGET_NAME ${TARGET_PATH} NAME_WE)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/${TARGET_PATH}
            DESTINATION "${CMAKE_BINARY_DIR}/pypi"
            PATTERN "*.egg-info" EXCLUDE
            PATTERN "unit" EXCLUDE
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