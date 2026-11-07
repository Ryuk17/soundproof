# Shared algorithm components used by the MNN-backed algorithms (KWS, ASR).
set(AE_ALGO_COMMON_SRC
    "${AE_ROOT}/feature/feature_extractor.cc"
    "${AE_ROOT}/utils/symbol_table.cc"
    "${AE_ROOT}/model/online_transducer_model.cc"
)

# ============================================================================
# Helper: register an interface test executable
#   usage: add_ae_test(<test_name> <test_source> <lib_target>)
# ============================================================================
macro(add_ae_test TEST_NAME TEST_SRC LIB_TARGET)
    add_executable(${TEST_NAME} "${AE_TESTS}/${TEST_SRC}")

    target_include_directories(${TEST_NAME} PRIVATE
        "${AE_ROOT}"
        "${AE_ROOT}/interface/"
        "${AE_ROOT}/utils/"
    )

    target_compile_definitions(${TEST_NAME} PRIVATE AUDIO_ENGINE_STATIC)

    if(WIN32 AND MINGW)
        target_link_libraries(${TEST_NAME} PRIVATE
            ${LIB_TARGET}
            absl::strings
            winmm
        )
    else()
        target_link_libraries(${TEST_NAME} PRIVATE
            ${LIB_TARGET}
            absl::strings
        )
    endif()

    # Tests read assets/... relative to the repo root
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
             WORKING_DIRECTORY "${AE_PROJECT_ROOT}")
endmacro()

# ============================================================================
# Common source groups (shared across multiple libraries)
# ============================================================================
set(DSP_COMMON
    "${AE_DSP}/spl_init.c"
    "${AE_DSP}/spl_inl.c"
    "${AE_DSP}/spl_sqrt.c"
    "${AE_DSP}/spl_sqrt_floor.c"
    "${AE_DSP}/copy_set_operations.c"
    "${AE_DSP}/dot_product_with_scale.cc"
    "${AE_DSP}/division_operations.c"
    "${AE_DSP}/min_max_operations.c"
    "${AE_DSP}/vector_operations.c"
    "${AE_DSP}/vector_scaling_operations.c"
)

set(DSP_RESAMPLE
    "${AE_DSP}/energy.c"
    "${AE_DSP}/resample.c"
    "${AE_DSP}/resample_fractional.c"
    "${AE_DSP}/resample_48khz.c"
    "${AE_DSP}/resample_by_2.c"
    "${AE_DSP}/resample_by_2_internal.c"
    "${AE_DSP}/get_scaling_square.c"
    "${AE_DSP}/downsample_fast.c"
    "${AE_DSP}/filter_ar.c"
)

set(NE10_FFT_SRC
    "${NEON_FFT_SOURCE_DIR}/src/NE10_fft.c"
    "${NEON_FFT_SOURCE_DIR}/src/NE10_fft_float32.c"
    "${NEON_FFT_SOURCE_DIR}/src/NE10_fft_generic_float32.c"
    "${NEON_FFT_SOURCE_DIR}/src/NE10_rfft_float32.c"
)

set(NE10_FFT_INCLUDE
    "${NEON_FFT_SOURCE_DIR}/include/"
)

# ---- Resampler sources (used by modules that need AudioBuffer) ----
file(GLOB RESAMPLER_SRC "${AE_CORE}/sample_rate_conversion/*.cc")
list(FILTER RESAMPLER_SRC EXCLUDE REGEX ".*_neon\\.cc$")

# sinc_resampler needs AlignedMalloc/AlignedFree (utils/aligned_malloc.cc)
set(RESAMPLER_DEPS "${AE_ROOT}/utils/aligned_malloc.cc")

# ---- Common include directories for libraries ----
set(LIB_COMMON_INCLUDES
    "${AE_ROOT}"
    "${AE_ROOT}/interface/"
    "${AE_ROOT}/utils/"
    "${AE_DSP}/"
)

# ---- Common utils ----
set(UTILS_WAV
    "${AE_ROOT}/utils/dr_wav.cc"
    "${AE_ROOT}/utils/audio_util.cc"
)

set(UTILS_AUDIO_BUFFER_AND_SPLITTING
    "${AE_ROOT}/utils/audio_buffer.cc"
    "${AE_DSP}/splitting_filter.cc"
    "${AE_DSP}/three_band_filter_bank.cc"
)
