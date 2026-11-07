# ============================================================================
# libAE_AECM + test_ae_aecm
# ============================================================================
    add_library(AE_AECM STATIC
        "${AE_ROOT}/interface/audio_engine_aecm.cpp"
        ${UTILS_WAV}
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        "${AE_DSP}/downsample_fast.c"
        "${AE_DSP}/randomization_functions.c"
        "${AE_DSP}/cross_correlation.c"
        "${AE_ROOT}/utils/ring_buffer.c"
        "${AE_DSP}/delay_estimator.cc"
        "${AE_DSP}/delay_estimator_wrapper.cc"
    )

    file(GLOB AECM_SRC "${AE_CORE}/acoustic_echo_cancellation_mobile/*.cc")
    list(FILTER AECM_SRC EXCLUDE REGEX ".*_mips\\.cc$")
    list(FILTER AECM_SRC EXCLUDE REGEX ".*_neon\\.cc$")
    target_sources(AE_AECM PRIVATE ${AECM_SRC})

    target_include_directories(AE_AECM PUBLIC
        ${LIB_COMMON_INCLUDES}
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_AECM PUBLIC absl::strings)
    target_compile_definitions(AE_AECM PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AECM PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_aecm "test_ae_aecm.cc" AE_AECM)
