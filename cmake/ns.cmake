# ============================================================================
# libAE_NS + test_ae_ns
# ============================================================================
    add_library(AE_NS STATIC
        "${AE_ROOT}/interface/audio_engine_ns.cpp"
        ${UTILS_WAV}
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
    )

    file(GLOB NS_SRC "${AE_CORE}/noise_suppressor/*.cc")
    target_sources(AE_NS PRIVATE ${NS_SRC} ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    target_include_directories(AE_NS PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_NS PUBLIC absl::strings)
    target_compile_definitions(AE_NS PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_NS PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_ns "test_ae_ns.cc" AE_NS)
