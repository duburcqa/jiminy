function(pythonDocstingSubstitution)
    # \brief    Substitute Docstring @copydoc flags with C++ Doxygen documentations.
    #
    # \details  Note that the dependency to each header must be specified explicitly
    #           to trigger the Docstring Substitution before build.
    #
    # \remark   ${CMAKE_BINARY_DIR}/${PROJECT_NAME} is the path of generated headers.

    file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
    FOREACH(header ${${PROJECT_NAME}_HEADERS})
        get_filename_component(header_name ${header} NAME_WE)
        file(RELATIVE_PATH header_path ${CMAKE_CURRENT_SOURCE_DIR} ${header})

        add_custom_command(
            OUTPUT  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${header_path}
            COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/doc_py/python_docstring_substitution.py
                    ${CMAKE_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${header_path}
                    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${header_path}
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${header_path}
            DEPENDS ${CMAKE_SOURCE_DIR}/build/doc_py/python_docstring_substitution.py
        )
        add_custom_target(docstringSubstitute_${header_name}
            DEPENDS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${header_path}
        )
        add_dependencies(${PROJECT_NAME} docstringSubstitute_${header_name})
    ENDFOREACH(header)
endfunction()