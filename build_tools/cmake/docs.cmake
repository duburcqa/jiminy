macro(create_component_docs)
    # Look for Doxygen and Sphinx utilities
    find_package(Doxygen COMPONENTS dot)
    find_package(Sphinx)
    if (NOT Sphinx_FOUND OR NOT Doxygen_FOUND)
        install(CODE "message(FATAL_ERROR \"Doxygen or Sphinx not available.\")"
                COMPONENT docs
                EXCLUDE_FROM_ALL)
        return()
    endif()

    # Define some environment variables
    set(DOXYFILE_IN ${CMAKE_SOURCE_DIR}/build_tools/docs/jiminy.doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_BINARY_DIR}/jiminy.doxyfile)
    set(DOXYGEN_INPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)

    # Find all the public headers
    get_target_property(CORE_PUBLIC_HEADER_DIR ${LIBRARY_NAME}_core INTERFACE_INCLUDE_DIRECTORIES)
    file(GLOB_RECURSE CORE_PUBLIC_HEADERS ${CORE_PUBLIC_HEADER_DIR}/*.h)

    # Replace variables inside @@ with the current values
    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

    # Only regenerate Doxygen when the Doxyfile or public headers change
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                       COMMAND ${CMAKE_COMMAND} -E env bash -c "\
                           ${DOXYGEN_EXECUTABLE} -b ${DOXYFILE_OUT} &> >(\
                           grep -Ei --line-buffered 'warning\|generating')"
                       WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                       DEPENDS ${CORE_PUBLIC_HEADERS}
                       MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                       COMMENT "Generating C++ documentation with Doxygen..."
                       VERBATIM)

    # Nice named target so we can run the job easily
    add_custom_target(doxygen DEPENDS ${DOXYGEN_INDEX_FILE})

    # Define some environment variables
    set(SPHINX_SOURCE ${CMAKE_SOURCE_DIR}/docs)
    set(SPHINX_BUILD ${SPHINX_SOURCE}/html)
    set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

    # Find all the docs rst files
    file(GLOB DOCS_FILES_ALL LIST_DIRECTORIES FALSE ${SPHINX_SOURCE}/*.*)
    file(GLOB_RECURSE DOCS_FILES_API ${SPHINX_SOURCE}/api/*.*)
    list(APPEND DOCS_FILES_ALL ${DOCS_FILES_API})

    # Only regenerate Sphinx when:
    # - Doxygen has rerun
    # - Doc files have been updated
    # - Sphinx config has been updated
    add_custom_command(OUTPUT ${SPHINX_INDEX_FILE} ALWAYS_REBUILD  # Dummy "file" to force systematc generation
                       COMMAND
                         ${SPHINX_EXECUTABLE} -b html
                         -Dbreathe_projects.${PROJECT_NAME}=${DOXYGEN_OUTPUT_DIR}/xml
                         ${SPHINX_SOURCE} ${SPHINX_BUILD}
                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                       DEPENDS ${DOXYGEN_INDEX_FILE} ${DOCS_FILES_ALL}
                       MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                       COMMENT "Generating unified documentation with Sphinx..."
                       VERBATIM)

    # Nice named target so we can run the job easily
    add_custom_target(sphinx DEPENDS ${DOXYGEN_INDEX_FILE} ${SPHINX_INDEX_FILE})

    # Install both documentations
    install(CODE "execute_process(COMMAND ${CMAKE_MAKE_PROGRAM} doxygen)
                  execute_process(COMMAND ${CMAKE_MAKE_PROGRAM} sphinx)"
            COMPONENT docs
            EXCLUDE_FROM_ALL)
ENDMACRO()
