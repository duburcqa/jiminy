# Import utils to define package version
include(CMakePackageConfigHelpers)

function(exportCmakeConfigFiles)
    # \brief    Export and install Cmake target configuration files.
    #
    # \details  Those configuration files can be used later to import
    #           a already compiled target in a convenient way using
    #           Cmake `find_package` command.

    set(ARGS ${ARGN})
    list(LENGTH ARGS NUM_ARGS)
    if(${NUM_ARGS} LESS 1)
        set(ARGS ${PROJECT_NAME})
    endif()

    export(TARGETS ${ARGS}
           FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    )
    install(EXPORT
            ${PROJECT_NAME}Config
            DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION ${BUILD_VERSION}
        COMPATIBILITY ${COMPATIBILITY_VERSION}
    )

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
    )
endfunction()

