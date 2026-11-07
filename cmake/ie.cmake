# ============================================================================
# libAE_IE + test_ae_ie
# ============================================================================
    add_library(AE_IE STATIC
        "${AE_ROOT}/interface/audio_engine_ie.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        "${AE_DSP}/randomization_functions.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        # IE core sources
        "${AE_CORE}/intelligibility_enhancer/intelligibility_enhancer.cc"
        "${AE_CORE}/intelligibility_enhancer/intelligibility_utils.cc"
        # Shared deps (same as BF: lapped_transform, real_fourier, window_generator, blocker)
        "${AE_DSP}/lapped_transform.cc"
        "${AE_DSP}/window_generator.cc"
        "${AE_ROOT}/utils/blocker.cc"
        "${AE_ROOT}/utils/ring_buffer.c"
        "${AE_ROOT}/utils/audio_ring_buffer.cc"
        # RealFourier (NE10-based R2C/C2R FFT)
        "${AE_DSP}/real_fourier.cc"
        "${AE_ROOT}/utils/aligned_malloc.cc"
    )

    target_sources(AE_IE PRIVATE ${RESAMPLER_SRC})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_IE PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/intelligibility_enhancer/"
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_IE PUBLIC absl::strings)
    target_compile_definitions(AE_IE PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_IE PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_ie "test_ae_ie.cc" AE_IE)
