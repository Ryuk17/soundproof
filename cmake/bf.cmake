# ============================================================================
# libAE_BF + test_ae_bf
# ============================================================================
    add_library(AE_BF STATIC
        "${AE_ROOT}/interface/audio_engine_bf.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        "${AE_DSP}/randomization_functions.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        # BF core sources
        "${AE_CORE}/beamforming/nonlinear_beamformer.cc"
        "${AE_CORE}/beamforming/array_util.cc"
        "${AE_CORE}/beamforming/covariance_matrix_generator.cc"
        # BF dependencies (common_audio)
        "${AE_DSP}/lapped_transform.cc"
        "${AE_DSP}/window_generator.cc"
        "${AE_ROOT}/utils/blocker.cc"
        "${AE_ROOT}/utils/ring_buffer.c"
        "${AE_ROOT}/utils/audio_ring_buffer.cc"
        # RealFourier (NE10-based R2C/C2R FFT)
        "${AE_DSP}/real_fourier.cc"
        "${AE_ROOT}/utils/aligned_malloc.cc"
    )

    target_sources(AE_BF PRIVATE ${RESAMPLER_SRC})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_BF PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/beamforming/"
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_BF PUBLIC absl::strings)
    target_compile_definitions(AE_BF PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_BF PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_bf "test_ae_bf.cc" AE_BF)
