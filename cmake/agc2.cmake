# ============================================================================
# libAE_AGC2 + test_ae_agc2
# ============================================================================
    add_library(AE_AGC2 STATIC
        "${AE_ROOT}/interface/audio_engine_agc2.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        "${RNN_VAD_DIR}/rnn_vad_weights.cc"
        "${AE_DSP}/pffft_wrapper.cc"
        "${PFFFT_SOURCE_DIR}/pffft.c"
        "${PFFFT_SOURCE_DIR}/fftpack.c"
    )

    file(GLOB AGC2_CORE "${AE_CORE}/automatic_gain_control2/*.cc")
    file(GLOB RNN_VAD_SRC "${AE_CORE}/automatic_gain_control2/rnn_vad/*.cc")
    target_sources(AE_AGC2 PRIVATE ${AGC2_CORE} ${RNN_VAD_SRC} ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/automatic_gain_control2/rnn_vad/vector_math_avx2.cc"
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_AGC2 PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
        "${RNN_VAD_DIR}"
        "${PFFFT_SOURCE_DIR}/"
        "${CMAKE_CURRENT_BINARY_DIR}/pffft_compat"
    )
    target_link_libraries(AE_AGC2 PUBLIC absl::strings)
    target_compile_definitions(AE_AGC2 PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AGC2 PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_agc2 "test_ae_agc2.cc" AE_AGC2)
