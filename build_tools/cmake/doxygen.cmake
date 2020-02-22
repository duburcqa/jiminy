MACRO (BUILD_DOC)
    # Build the doc
    # =============

    find_package(Doxygen QUIET COMPONENTS dot)

    # Build the C++ documentation.
    configure_file(${CMAKE_SOURCE_DIR}/build_tools/doc_cpp/jiminy.doxyfile
    ${CMAKE_BINARY_DIR}/jiminy.doxyfile @ONLY)
    add_custom_target(doc
        COMMAND ${CMAKE_COMMAND} -E env bash -c "\
            doxygen -b ${CMAKE_BINARY_DIR}/jiminy.doxyfile &> >(\
            grep -Ei --line-buffered 'warning\|generating')"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen..."
        VERBATIM
    )

    # Build the Python documentation.
    configure_file(${CMAKE_SOURCE_DIR}/build_tools/doc_py/jiminy_python.doxyfile
    ${CMAKE_BINARY_DIR}/jiminy_python.doxyfile @ONLY)
    add_custom_target(doc_py
        COMMAND ${CMAKE_COMMAND} -E env bash -c "\
            doxygen -b ${CMAKE_BINARY_DIR}/jiminy_python.doxyfile &> >(\
            grep -Ei --line-buffered 'warning\|generating')"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating Python documentation with Doxygen..."
        VERBATIM
    )

    # Install both documentations (if doxygen is available).
    if(${Doxygen_FOUND})
        install(
            CODE "execute_process (COMMAND ${CMAKE_MAKE_PROGRAM} doc)
                execute_process (COMMAND ${CMAKE_MAKE_PROGRAM} doc_py)
                file (REMOVE_RECURSE \"${CMAKE_SOURCE_DIR}/docs/cpp\" \"${CMAKE_SOURCE_DIR}/docs/python\")
                execute_process (COMMAND ${CMAKE_COMMAND} -E copy_directory \"${CMAKE_BINARY_DIR}/doc/html/\" \"${CMAKE_SOURCE_DIR}/docs/cpp\"
                                COMMAND ${CMAKE_COMMAND} -E copy_directory \"${CMAKE_BINARY_DIR}/doc_py/html/\" \"${CMAKE_SOURCE_DIR}/docs/python\"
                                )"
            COMPONENT doxygen
            EXCLUDE_FROM_ALL
        )
    else(${Doxygen_FOUND})
        message("-- Doxygen with Dot component not found. Documentation generation disabled.")
    endif(${Doxygen_FOUND})
ENDMACRO()
