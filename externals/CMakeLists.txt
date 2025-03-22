message("[aloe] --- Adding external dependencies ---")

# If we are running tests, include gtest/gmock
if (ALOE_ENABLE_TESTS)
    message("++++++ GoogleTest and GoogleMock (Testing Frameworks)")
    add_subdirectory(googletest EXCLUDE_FROM_ALL)

    set_target_properties(gtest PROPERTIES FOLDER "Externals/")
    set_target_properties(gtest_main PROPERTIES FOLDER "Externals/")
    set_target_properties(gmock PROPERTIES FOLDER "Externals/")
    set_target_properties(gmock_main PROPERTIES FOLDER "Externals/")
endif ()


# Add a descriptive footer message
message("[aloe] --- Finished adding external dependencies ---")