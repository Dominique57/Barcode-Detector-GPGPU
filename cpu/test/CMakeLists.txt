find_package(GTest)

if (GTest_FOUND)

    # gather all *.cc files
    file(GLOB TEST_SRC *.cc)

    # if you need to copy from source to build folder do it here
    # configure_file(SRC DST COPYONLY)

    # Dont create a unit_test if not tests given
    ADD_EXECUTABLE(UNIT_TEST ${TEST_SRC})

    TARGET_LINK_LIBRARIES(
        UNIT_TEST
        SOURCE_LIB
        GTest::GTest
    )

    add_test(NAME test_unitaire
             COMMAND UNIT_TEST)
endif()
