# ============================================================================
# libAE_TS + test_ae_ts
# ============================================================================
    add_library(AE_TS STATIC
        "${AE_ROOT}/interface/audio_engine_ts.cpp"
        ${UTILS_WAV}
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        "${AE_DSP}/randomization_functions.c"
        # TS core sources
        "${AE_CORE}/transient_suppressor/transient_suppressor.cc"
        "${AE_CORE}/transient_suppressor/transient_detector.cc"
        "${AE_CORE}/transient_suppressor/moving_moments.cc"
        "${AE_CORE}/transient_suppressor/wpd_node.cc"
        "${AE_CORE}/transient_suppressor/wpd_tree.cc"
        # FIR filter (plain C implementation)
        "${AE_ROOT}/utils/fir_filter.cc"
    )

    target_include_directories(AE_TS PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/transient_suppressor/"
        "${AE_CORE}/noise_suppressor/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_TS PUBLIC absl::strings)
    target_compile_definitions(AE_TS PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_TS PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_ts "test_ae_ts.cc" AE_TS)
