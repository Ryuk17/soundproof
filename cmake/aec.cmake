# ============================================================================
# libAE_AEC + test_ae_aec
# ============================================================================
    add_library(AE_AEC STATIC
        "${AE_ROOT}/interface/audio_engine_aec.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/echo_canceller3_config.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        "${AE_DSP}/high_pass_filter.cc"
        "${AE_DSP}/cascaded_biquad_filter.cc"
    )

    file(GLOB AEC3_SRC "${AE_CORE}/acoustic_echo_cancellation3/*.cc")
    list(FILTER AEC3_SRC EXCLUDE REGEX ".*neural_feature_extractor\\.cc$")
    list(FILTER AEC3_SRC EXCLUDE REGEX ".*neural_residual_echo_estimator_impl\\.cc$")
    target_sources(AE_AEC PRIVATE ${AEC3_SRC} ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/acoustic_echo_cancellation3/adaptive_fir_filter_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/adaptive_fir_filter_erl_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/fft_data_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/matched_filter_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/vector_math_avx2.cc"
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_AEC PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_AEC PUBLIC absl::strings)
    target_compile_definitions(AE_AEC PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AEC PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_aec "test_ae_aec.cc" AE_AEC)
