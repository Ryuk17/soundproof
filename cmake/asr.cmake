# ============================================================================
# libAE_ASR + test_ae_asr — streaming ASR with pluggable model backends
# (transducer: zipformer/zipformer2/conformer/lstm; paraformer).
# Requires MNN.
# ============================================================================
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_ASR=ON but WITH_MNN=OFF; skipping libAE_ASR")
    else()
        include(${AE_PROJECT_ROOT}/cmake/knf.cmake)
        include(${AE_PROJECT_ROOT}/cmake/ssentencepiece.cmake)

        set(ASR_CORE_SRC
            "${AE_CORE}/speech_recognition/asr_backend_factory.cc"
            "${AE_CORE}/speech_recognition/transducer_backend.cc"
            "${AE_CORE}/speech_recognition/transducer_decoder.cc"
            "${AE_CORE}/speech_recognition/transducer_beam_search_decoder.cc"
            "${AE_CORE}/speech_recognition/hypothesis.cc"
            "${AE_CORE}/speech_recognition/online_rnn_lm.cc"
            "${AE_CORE}/speech_recognition/punctuation_model.cc"
            "${AE_CORE}/speech_recognition/online_punctuator.cc"
            "${AE_CORE}/speech_recognition/paraformer_model.cc"
            "${AE_CORE}/speech_recognition/paraformer_backend.cc"
            "${AE_CORE}/speech_recognition/paraformer_utils.cc"
            "${AE_CORE}/speech_recognition/endpoint.cc"
            "${AE_CORE}/speech_recognition/online_recognizer.cc"
        )

        add_library(AE_ASR STATIC
            "${AE_ROOT}/interface/audio_engine_asr.cpp"
            "${AE_ROOT}/utils/dr_wav.cc"
            ${ASR_CORE_SRC}
            ${AE_ALGO_COMMON_SRC}
            ${AE_MNN_ENGINE_SRC}
        )

        target_include_directories(AE_ASR PUBLIC
            ${LIB_COMMON_INCLUDES}
            "${AE_CORE}/speech_recognition"
        )
        target_link_libraries(AE_ASR PRIVATE
            soundproof_mnn
            kaldi-native-fbank-core
            ssentencepiece_core
        )
        target_compile_definitions(AE_ASR PRIVATE AUDIO_ENGINE_EXPORTS)
        set_target_properties(AE_ASR PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

        add_ae_test(test_ae_asr "test_ae_asr.cc" AE_ASR)
    endif()
