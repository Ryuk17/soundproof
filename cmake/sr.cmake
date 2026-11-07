# ============================================================================
# libAE_SR + test_ae_sr — speaker recognition (identification/verification)
# with pluggable embedding backends (wespeaker/3d-speaker/nemo).
# Requires MNN.
# ============================================================================
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_SR=ON but WITH_MNN=OFF; skipping libAE_SR")
    else()
        include(${AE_PROJECT_ROOT}/cmake/knf.cmake)

        set(SR_CORE_SRC
            "${AE_CORE}/speaker_recognition/sr_backend_factory.cc"
            "${AE_CORE}/speaker_recognition/general_sr_backend.cc"
            "${AE_CORE}/speaker_recognition/nemo_sr_backend.cc"
            "${AE_CORE}/speaker_recognition/speaker_embedding_manager.cc"
            "${AE_CORE}/speaker_recognition/speaker_recognizer.cc"
        )

        add_library(AE_SR STATIC
            "${AE_ROOT}/interface/audio_engine_sr.cpp"
            "${AE_ROOT}/utils/dr_wav.cc"
            "${AE_ROOT}/feature/feature_extractor.cc"
            ${SR_CORE_SRC}
            ${AE_MNN_ENGINE_SRC}
        )

        target_include_directories(AE_SR PUBLIC
            ${LIB_COMMON_INCLUDES}
            "${AE_CORE}/speaker_recognition"
        )
        target_link_libraries(AE_SR PRIVATE
            soundproof_mnn
            kaldi-native-fbank-core
        )
        target_compile_definitions(AE_SR PRIVATE AUDIO_ENGINE_EXPORTS)
        set_target_properties(AE_SR PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

        add_ae_test(test_ae_sr "test_ae_sr.cc" AE_SR)
    endif()
