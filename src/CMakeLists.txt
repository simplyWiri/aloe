# To make the actual definition of the library a little less - "scatter brained" we instead make a helper function to
# abstract all logic regarding exporting / library creation.
# Define a helper function to simplify adding libraries.
function(aloe_add_library libname)
    cmake_parse_arguments(ALC "" "" "SOURCES;HEADERS;LINK_AGAINST" ${ARGN})

    add_library(${libname})
    target_sources(${libname} PRIVATE ${ALC_SOURCES})
    target_include_directories(${libname} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
            $<INSTALL_INTERFACE:include>
    )
    if(ALC_LINK_AGAINST)
        target_link_libraries(${libname} PUBLIC ${ALC_LINK_AGAINST})
    endif()

    # Automatically prepend `${CMAKE_CURRENT_SOURCE_DIR}/../include/aloe/` to each header listed.
    set(public_headers "")
    foreach(header IN LISTS ALC_HEADERS)
        list(APPEND public_headers "${CMAKE_CURRENT_SOURCE_DIR}/../include/aloe/${header}")
    endforeach()

    set_target_properties(${libname} PROPERTIES PUBLIC_HEADER "${public_headers}")

    include(GNUInstallDirs)
    install(TARGETS ${libname}
            EXPORT ${libname}Targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${libname}
    )

    install(EXPORT ${libname}Targets
            FILE ${libname}Targets.cmake
            NAMESPACE ${libname}::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${libname}
    )
endfunction()

aloe_add_library(aloe
    HEADERS
        core/application.h
    SOURCES
        core/application.cpp
)