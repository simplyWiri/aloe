message("[aloe] --- Adding external dependencies ---")

message("++++++ tl::expected (Functional style error handling)")
set(EXPECTED_BUILD_TESTS)
add_subdirectory(expected EXCLUDE_FROM_ALL)

message("++++++ Volk (Meta loader for Vulkan functions)")
add_subdirectory(volk EXCLUDE_FROM_ALL)

message("++++++ GFLW (Cross platformw window management)")
add_subdirectory(glfw EXCLUDE_FROM_ALL)

message("++++++ VMA (GPU Memory allocator)")
add_subdirectory(VulkanMemoryAllocator EXCLUDE_FROM_ALL)

# If we are running tests, include gtest/gmock
if (ALOE_ENABLE_TESTS)
    message("++++++ GoogleTest and GoogleMock (Testing Frameworks)")
    add_subdirectory(googletest EXCLUDE_FROM_ALL)

    set_target_properties(gtest PROPERTIES FOLDER "externals/")
    set_target_properties(gtest_main PROPERTIES FOLDER "externals/")
    set_target_properties(gmock PROPERTIES FOLDER "externals/")
    set_target_properties(gmock_main PROPERTIES FOLDER "externals/")
endif ()

set_target_properties(volk PROPERTIES FOLDER "externals/")
set_target_properties(VulkanMemoryAllocator PROPERTIES FOLDER "externals/")


# Add a descriptive footer message
message("[aloe] --- Finished adding external dependencies ---")