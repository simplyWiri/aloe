# Dummy test executable
add_executable(basic_test test.cpp)
target_link_libraries(basic_test gtest gtest_main aloe)

# Add the test to CTest
add_test(NAME BasicTest COMMAND basic_test)