# ============================================================================
# test_mnn_engine — only when WITH_MNN=ON. The engine sources are compiled
# straight into the test executable (no standalone library).
# ============================================================================
    add_executable(test_mnn_engine
        "${AE_TESTS}/test_mnn_engine.cc"
        ${AE_MNN_ENGINE_SRC}
    )

    target_include_directories(test_mnn_engine PRIVATE
        ${LIB_COMMON_INCLUDES}
        "${AE_ROOT}/mnn_engine"
    )

    target_link_libraries(test_mnn_engine PRIVATE soundproof_mnn)
    if(WIN32 AND MINGW)
        target_link_libraries(test_mnn_engine PRIVATE winmm)
    endif()

    add_test(NAME test_mnn_engine COMMAND test_mnn_engine
             WORKING_DIRECTORY "${AE_PROJECT_ROOT}")
