# ============================================================================
# libAE_HS + test_ae_hs
# ============================================================================
    add_library(AE_HS STATIC
        "${AE_ROOT}/interface/audio_engine_hs.cpp"
        ${UTILS_WAV}
        "${AE_CORE}/howling_suppressor/howling_core.c"
        "${AE_CORE}/howling_suppressor/biquad.c"
        "${AE_CORE}/howling_suppressor/tone_detect.c"
    )

    target_include_directories(AE_HS PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/howling_suppressor/"
    )
    target_link_libraries(AE_HS PUBLIC absl::strings)
    target_compile_definitions(AE_HS PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_HS PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_hs "test_ae_hs.cc" AE_HS)
