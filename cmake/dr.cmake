# ============================================================================
# libAE_DR + test_ae_dr
# ============================================================================
# Eigen (header-only, double-precision FFT required by the dereverberation
# EKF — kept on purpose, see real_fft.h). Fetched above when not provided;
# override with -DEIGEN3_ROOT=<eigen source root>.
    find_path(EIGEN3_INCLUDE_DIR "Eigen/Dense"
        HINTS ${EIGEN3_ROOT} $ENV{EIGEN3_ROOT}
        NO_DEFAULT_PATH
    )
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen not found (BUILD_AE_DR=ON). "
                "Set -DEIGEN3_ROOT=<eigen source root> to point at Eigen.")
    endif()
    message(STATUS "EIGEN3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")

    add_library(AE_DR STATIC
        "${AE_ROOT}/interface/audio_engine_dr.cpp"
        ${UTILS_WAV}
        "${AE_CORE}/dereverberation/dereverberation.cc"
        "${AE_CORE}/dereverberation/gaussian_likelihood.cc"
        "${AE_CORE}/dereverberation/hmm_dictionary.cc"
        "${AE_CORE}/dereverberation/mel_filterbank.cc"
        "${AE_CORE}/dereverberation/real_fft.cc"
    )

    target_include_directories(AE_DR PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/"
        ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(AE_DR PUBLIC absl::strings)
    target_compile_definitions(AE_DR PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_DR PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_dr "test_ae_dr.cc" AE_DR)
