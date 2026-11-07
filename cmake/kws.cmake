# ============================================================================
# libAE_KWS + test_ae_kws — zipformer2 transducer keyword spotting.
# Requires MNN (the model is executed through the internal mnn_engine).
# ============================================================================
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_KWS=ON but WITH_MNN=OFF; skipping libAE_KWS")
    else()
        include(${AE_PROJECT_ROOT}/cmake/knf.cmake)

        set(KWS_CORE_SRC
            "${AE_CORE}/keywords_spotting/context_graph.cc"
            "${AE_CORE}/keywords_spotting/keyword_encoder.cc"
            "${AE_CORE}/keywords_spotting/keyword_decoder.cc"
            "${AE_CORE}/keywords_spotting/keyword_spotter.cc"
        )

        add_library(AE_KWS STATIC
            "${AE_ROOT}/interface/audio_engine_kws.cpp"
            "${AE_ROOT}/utils/dr_wav.cc"
            ${KWS_CORE_SRC}
            ${AE_ALGO_COMMON_SRC}
            ${AE_MNN_ENGINE_SRC}
        )

        target_include_directories(AE_KWS PUBLIC
            ${LIB_COMMON_INCLUDES}
            "${AE_CORE}/keywords_spotting"
        )
        target_link_libraries(AE_KWS PRIVATE
            soundproof_mnn
            kaldi-native-fbank-core
        )
        target_compile_definitions(AE_KWS PRIVATE AUDIO_ENGINE_EXPORTS)
        set_target_properties(AE_KWS PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

        add_ae_test(test_ae_kws "test_ae_kws.cc" AE_KWS)
    endif()
