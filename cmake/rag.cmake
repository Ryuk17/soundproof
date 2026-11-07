# ============================================================================
# libAE_RAG + test_ae_rag — RAG text embeddings via MNN-LLM
# (Qwen3-Embedding). Requires WITH_MNN_LLM.
# ============================================================================
    if(NOT WITH_MNN_LLM)
        message(STATUS "BUILD_AE_RAG=ON but WITH_MNN_LLM=OFF; skipping libAE_RAG")
    else()
        set(RAG_CORE_SRC
            "${AE_CORE}/retrieval_augmented_generation/embedding_backend_factory.cc"
            "${AE_CORE}/retrieval_augmented_generation/mnn_llm_embedding_backend.cc"
            "${AE_CORE}/retrieval_augmented_generation/embedding_engine.cc"
        )

        add_library(AE_RAG STATIC
            "${AE_ROOT}/interface/audio_engine_rag.cpp"
            ${RAG_CORE_SRC}
            ${AE_MNN_LLM_SRC}
        )

        target_include_directories(AE_RAG PUBLIC
            ${LIB_COMMON_INCLUDES}
            "${AE_CORE}/retrieval_augmented_generation"
        )
        target_link_libraries(AE_RAG PRIVATE soundproof_mnn)
        target_compile_definitions(AE_RAG PRIVATE AUDIO_ENGINE_EXPORTS)
        set_target_properties(AE_RAG PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

        add_ae_test(test_ae_rag "test_ae_rag.cc" AE_RAG)
    endif()
