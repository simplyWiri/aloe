#include <aloe/core/application.h>
#include <gtest/gtest.h>

// An example test case
TEST(HelloWorldTest, BasicAssertion) {
    aloe::Application application;

    EXPECT_EQ(application.name, "aloe application");
}
