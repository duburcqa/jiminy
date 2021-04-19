function(pythonDocstingSubstitution)
    # \brief    Substitute Docstring @copydoc flags with C++ Doxygen documentations.
    #
    # \details  Note that the dependency to each file must be specified explicitly
    #           to trigger the Docstring Substitution before build.
    #
    # \remark   ${CMAKE_CURRENT_BINARY_DIR} is the path of generated files.

    file(GLOB_RECURSE ${PROJECT_NAME}_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
    FOREACH(file ${${PROJECT_NAME}_FILES})
        get_filename_component(file_name ${file} NAME_WE)
        file(RELATIVE_PATH file_path ${CMAKE_CURRENT_SOURCE_DIR} ${file})

        add_custom_command(
            OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/${file_path}
            COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/docs/python_docstring_substitution.py
                    ${CMAKE_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${file_path}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file_path}
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${file_path}
            DEPENDS ${CMAKE_SOURCE_DIR}/build_tools/docs/python_docstring_substitution.py
        )
        string(RANDOM target_uid)
        set(target_name docstringSubstitute_${file_name}_${target_uid})
        add_custom_target(${target_name}
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file_path}
        )
        add_dependencies(${PROJECT_NAME} ${target_name})
    ENDFOREACH(file)
endfunction()
