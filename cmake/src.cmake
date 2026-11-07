# ============================================================================
# libAE_SRC + test_ae_src
# ============================================================================
    add_library(AE_SRC STATIC
        "${AE_ROOT}/interface/audio_engine_src.cpp"
        ${UTILS_WAV}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
    )

    target_sources(AE_SRC PRIVATE ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_SRC PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
    )
    target_link_libraries(AE_SRC PUBLIC absl::strings)
    target_compile_definitions(AE_SRC PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_SRC PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_src "test_ae_src.cc" AE_SRC)
