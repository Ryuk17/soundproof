# ============================================================================
# libAE_TURN + test_ae_turn — Smart Turn v3.2 end-of-turn detection.
# Requires MNN.
# ============================================================================
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_TURN=ON but WITH_MNN=OFF; skipping libAE_TURN")
    else()
        set(TURN_CORE_SRC
            "${AE_CORE}/turn_detection/turn_backend_factory.cc"
            "${AE_CORE}/turn_detection/smart_turn_backend.cc"
            "${AE_ROOT}/feature/whisper_mel.cc"
            "${AE_CORE}/turn_detection/turn_detector.cc"
        )

        add_library(AE_TURN STATIC
            "${AE_ROOT}/interface/audio_engine_turn.cpp"
            "${AE_ROOT}/utils/dr_wav.cc"
            "${AE_DSP}/pffft_wrapper.cc"
            "${PFFFT_SOURCE_DIR}/pffft.c"
            "${PFFFT_SOURCE_DIR}/fftpack.c"
            ${TURN_CORE_SRC}
            ${AE_MNN_ENGINE_SRC}
        )

        target_include_directories(AE_TURN PUBLIC
            ${LIB_COMMON_INCLUDES}
            "${AE_CORE}/turn_detection"
            "${PFFFT_SOURCE_DIR}/"
            "${CMAKE_CURRENT_BINARY_DIR}/pffft_compat"
        )
        target_link_libraries(AE_TURN PRIVATE soundproof_mnn)
        target_compile_definitions(AE_TURN PRIVATE AUDIO_ENGINE_EXPORTS)
        set_target_properties(AE_TURN PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

        add_ae_test(test_ae_turn "test_ae_turn.cc" AE_TURN)
    endif()
