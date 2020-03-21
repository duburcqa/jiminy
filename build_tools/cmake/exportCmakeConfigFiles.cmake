# Import utils to define package version
include(CMakePackageConfigHelpers)

function(exportCmakeConfigFiles)
    # \brief    Export and install Cmake target configuration files.
    #
    # \details  Those configuration files can be used later to import
    #           a already compiled target in a convenient way using
    #           Cmake `find_package` command.

    export(TARGETS ${PROJECT_NAME}
           FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    )
    install(EXPORT
            ${PROJECT_NAME}Config
            DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
    )

    if(${CMAKE_VERSION} VERSION_GREATER "3.11.0")
        set(COMPATIBILITY_VERSION SameMinorVersion)
    else(${CMAKE_VERSION} VERSION_GREATER "3.11.0")
        set(COMPATIBILITY_VERSION AnyNewerVersion)
    endif()

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


