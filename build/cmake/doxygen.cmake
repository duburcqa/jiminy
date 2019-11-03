MACRO (BUILD_DOC)
	# Build the doc
	# =============

	# Build the C++ documentation.
	configure_file (${CMAKE_SOURCE_DIR}/build/doc_cpp/jiminy.doxyfile
    ${CMAKE_BINARY_DIR}/jiminy.doxyfile @ONLY)
	add_custom_target (doc
		COMMAND ${CMAKE_COMMAND} -E env ${BASH} -c "\
			doxygen -b ${CMAKE_BINARY_DIR}/jiminy.doxyfile &> >(\
			grep -Ei --line-buffered 'warning\|generating')"
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		COMMENT "Generating documentation with Doxygen..."
		EXCLUDE_FROM_ALL TRUE
		VERBATIM)

	# Build the Python documentation.
	configure_file (${CMAKE_SOURCE_DIR}/build/doc_py/jiminy_python.doxyfile
    ${CMAKE_BINARY_DIR}/jiminy_python.doxyfile @ONLY)
	add_custom_target (doc_py
		COMMAND ${CMAKE_COMMAND} -E env ${BASH} -c "\
			doxygen -b ${CMAKE_BINARY_DIR}/jiminy_python.doxyfile &> >(\
			grep -Ei --line-buffered 'warning\|generating')"
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMENT "Generating Python documentation with Doxygen..."
		EXCLUDE_FROM_ALL TRUE
		VERBATIM)

ENDMACRO()
