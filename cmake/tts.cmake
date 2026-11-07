# ============================================================================
# libAE_TTS + test_ae_tts — text-to-speech with pluggable model backends
# (Kokoro, VITS/Piper). Requires MNN.
# ============================================================================
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_TTS=ON but WITH_MNN=OFF; skipping libAE_TTS")
    else()
        set(TTS_CORE_SRC
            "${AE_CORE}/text_to_speech/tts_tokenizer.cc"
            "${AE_CORE}/text_to_speech/tts_backend_factory.cc"
            "${AE_CORE}/text_to_speech/kokoro_backend.cc"
            "${AE_CORE}/text_to_speech/vits_backend.cc"
            "${AE_CORE}/text_to_speech/tts_engine.cc"
        )

        add_library(AE_TTS STATIC
            "${AE_ROOT}/interface/audio_engine_tts.cpp"
            ${TTS_CORE_SRC}
            ${AE_MNN_ENGINE_SRC}
        )

        target_include_directories(AE_TTS PUBLIC
            ${LIB_COMMON_INCLUDES}
            "${AE_CORE}/text_to_speech"
        )
        target_link_libraries(AE_TTS PRIVATE soundproof_mnn)
        target_compile_definitions(AE_TTS PRIVATE AUDIO_ENGINE_EXPORTS)
        set_target_properties(AE_TTS PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

        add_ae_test(test_ae_tts "test_ae_tts.cc" AE_TTS)
    endif()
