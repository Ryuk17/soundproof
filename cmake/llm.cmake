# ============================================================================
# libAE_LLM + test_ae_llm — Qwen3.5 VLM chat via MNN-LLM.
# Requires WITH_MNN_VISION.
# ============================================================================
    if(NOT WITH_MNN_VISION)
        message(STATUS "BUILD_AE_LLM=ON but WITH_MNN_VISION=OFF; skipping libAE_LLM")
    else()
        set(LLM_CORE_SRC
            "${AE_CORE}/llm/llm_backend_factory.cc"
            "${AE_CORE}/llm/mnn_llm_backend.cc"
            "${AE_CORE}/llm/llm_engine.cc"
        )

        add_library(AE_LLM STATIC
            "${AE_ROOT}/interface/audio_engine_llm.cpp"
            ${LLM_CORE_SRC}
            ${AE_MNN_LLM_SRC}
        )

        target_include_directories(AE_LLM PUBLIC
            ${LIB_COMMON_INCLUDES}
            "${AE_CORE}/llm"
        )
        target_link_libraries(AE_LLM PRIVATE soundproof_mnn)
        target_compile_definitions(AE_LLM PRIVATE AUDIO_ENGINE_EXPORTS)
        set_target_properties(AE_LLM PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

        add_ae_test(test_ae_llm "test_ae_llm.cc" AE_LLM)
    endif()
