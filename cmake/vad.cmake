# ============================================================================
# libAE_VAD + test_ae_vad
# ============================================================================
    add_library(AE_VAD STATIC
        "${AE_ROOT}/interface/audio_engine_vad.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        "${AE_DSP}/randomization_functions.c"
        "${AE_DSP}/cross_correlation.c"
        "${AE_CORE}/voice_activity_detection/vad_core/webrtc_vad.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_core.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_filterbank.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_gmm.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_sp.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
    )

    file(GLOB VAD_SRC "${AE_CORE}/voice_activity_detection/*.cc")
    # silero_vad.cc depends on MNN; it is added explicitly below when
    # WITH_MNN=ON so the default DSP-only build stays MNN-free.
    list(FILTER VAD_SRC EXCLUDE REGEX "silero_vad\\.cc$")
    file(GLOB ISAC_VAD_SRC "${AE_CORE}/voice_activity_detection/isac/main/source/*.c")
    target_sources(AE_VAD PRIVATE ${VAD_SRC} ${ISAC_VAD_SRC} ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_VAD PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_VAD PUBLIC absl::strings)
    target_compile_definitions(AE_VAD PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_VAD PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    if(WITH_MNN)
        # Silero VAD runs through the internal MNN engine; both are compiled
        # straight into AE_VAD (no standalone libraries).
        target_sources(AE_VAD PRIVATE
            "${AE_CORE}/voice_activity_detection/silero_vad.cc"
            ${AE_MNN_ENGINE_SRC}
        )
        target_include_directories(AE_VAD PRIVATE "${AE_ROOT}/mnn_engine")
        target_link_libraries(AE_VAD PRIVATE soundproof_mnn)
        target_compile_definitions(AE_VAD PRIVATE AE_WITH_MNN=1)
    endif()

    add_ae_test(test_ae_vad "test_ae_vad.cc" AE_VAD)
    if(WITH_MNN)
        target_compile_definitions(test_ae_vad PRIVATE AE_WITH_MNN=1)
    endif()
