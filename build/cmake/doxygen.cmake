MACRO (BUILD_DOC)
	# Build the doc
	# =============
	configure_file (${CMAKE_SOURCE_DIR}/build/doc_cpp/jiminy.doxyfile
    ${CMAKE_BINARY_DIR}/jiminy.doxyfile @ONLY)
	add_custom_target (doc
		COMMAND doxygen ${CMAKE_BINARY_DIR}/jiminy.doxyfile
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		COMMENT "Generating documentation with Doxygen..."
		EXCLUDE_FROM_ALL TRUE)
	# Python documentation.
	configure_file (${CMAKE_SOURCE_DIR}/build/doc_py/jiminy_python.doxyfile
    ${CMAKE_BINARY_DIR}/jiminy_python.doxyfile @ONLY)
	add_custom_target (doc_py
		COMMAND doxygen ${CMAKE_BINARY_DIR}/jiminy_python.doxyfile
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		COMMENT "Generating python documentation with Doxygen..."
		EXCLUDE_FROM_ALL TRUE)
ENDMACRO()
