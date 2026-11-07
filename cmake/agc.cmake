# ============================================================================
# libAE_AGC + test_ae_agc
# ============================================================================
    add_library(AE_AGC STATIC
        "${AE_ROOT}/interface/audio_engine_agc.cpp"
        ${UTILS_WAV}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
    )

    file(GLOB AGC_SRC "${AE_CORE}/automatic_gain_control/*.cc")
    target_sources(AE_AGC PRIVATE ${AGC_SRC})

    target_include_directories(AE_AGC PUBLIC ${LIB_COMMON_INCLUDES})
    target_link_libraries(AE_AGC PUBLIC absl::strings)
    target_compile_definitions(AE_AGC PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AGC PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_agc "test_ae_agc.cc" AE_AGC)
