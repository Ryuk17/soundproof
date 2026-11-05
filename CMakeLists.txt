# ============================================================================
# Soundproof — audio engine
#
#   Standalone build: third-party dependencies (abseil-cpp, neon-fft, pffft,
#   eigen) are fetched via FetchContent (URL tarball + SHA256 checksum).
#   A super build may provide them instead:
#     absl::strings target, NEON_FFT_SOURCE_DIR, PFFFT_SOURCE_DIR, EIGEN3_ROOT
#
#   Optional: the MNN inference runtime is fetched only with -DWITH_MNN=ON.
#
#   Build (Windows, MinGW):
#     cmake -S . -B build -G "MinGW Makefiles" \
#           -DCMAKE_TOOLCHAIN_FILE=toolchains/x86_64-windows.cmake
#     cmake --build build -j
# ============================================================================
cmake_minimum_required(VERSION 3.16)

project(SoundProofProject LANGUAGES C CXX)

if(CMAKE_TOOLCHAIN_FILE)
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_CURRENT_LIST_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

# Project root — CMAKE_CURRENT_LIST_DIR works both standalone and via
# add_subdirectory from a super build (PROJECT_SOURCE_DIR would not).
set(AE_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(LIBRARY_OUTPUT_PATH ${AE_PROJECT_ROOT}/lib)
set(EXECUTABLE_OUTPUT_PATH ${AE_PROJECT_ROOT}/bin)

# ---- Source layout (restructured from src/audio_engine) ----
set(AE_ROOT  "${AE_PROJECT_ROOT}/soundproof")   # interface/ utils/ dsp/ core/
set(AE_DSP   "${AE_ROOT}/dsp")                  # was signal_processing
set(AE_CORE  "${AE_ROOT}/core")                 # was audio_processing
set(AE_TESTS "${AE_PROJECT_ROOT}/tests")        # was unitest
set(RNN_VAD_DIR "${AE_PROJECT_ROOT}/models/rnnoise/src")  # was model_weights/rnnoise/src

# Shared algorithm components used by the MNN-backed algorithms (KWS, ASR).
set(AE_ALGO_COMMON_SRC
    "${AE_ROOT}/feature/feature_extractor.cc"
    "${AE_ROOT}/utils/symbol_table.cc"
    "${AE_ROOT}/model/online_transducer_model.cc"
)

# ============================================================================
# Build options — one per module (controls lib + interface test)
# ============================================================================
option(BUILD_AE_VAD  "Build libAE_VAD and test_ae_vad"   ON)
option(BUILD_AE_SRC  "Build libAE_SRC and test_ae_src"   ON)
option(BUILD_AE_NS   "Build libAE_NS and test_ae_ns"     ON)
option(BUILD_AE_AGC  "Build libAE_AGC and test_ae_agc"   ON)
option(BUILD_AE_AGC2 "Build libAE_AGC2 and test_ae_agc2" ON)
option(BUILD_AE_AECM "Build libAE_AECM and test_ae_aecm" ON)
option(BUILD_AE_AEC  "Build libAE_AEC and test_ae_aec"   ON)
option(BUILD_AE_BF   "Build libAE_BF and test_ae_bf"     ON)
option(BUILD_AE_HS   "Build libAE_HS and test_ae_hs"     ON)
option(BUILD_AE_IE   "Build libAE_IE and test_ae_ie"     ON)
option(BUILD_AE_TS   "Build libAE_TS and test_ae_ts"     ON)
option(BUILD_AE_DR   "Build libAE_DR and test_ae_dr"     ON)
option(BUILD_AE_KWS  "Build libAE_KWS and test_ae_kws (requires WITH_MNN)" ON)
option(BUILD_AE_ASR  "Build libAE_ASR and test_ae_asr (requires WITH_MNN)" ON)
option(BUILD_AE_SR   "Build libAE_SR and test_ae_sr (requires WITH_MNN)"   ON)

# Optional deep-learning runtime (MNN). OFF by default: heavy dependency
# (static lib ~12 MB, several minutes to build) used by future DNN modules.
option(WITH_MNN     "Build the MNN inference runtime"              OFF)
option(WITH_MNN_LLM "Build MNN with LLM support (implies WITH_MNN)" OFF)
if(WITH_MNN_LLM)
    set(WITH_MNN ON)
endif()

# ============================================================================
# Third-party dependencies — FetchContent when not provided by a super build
# ============================================================================
include(FetchContent)

set(ABSEIL_VERSION "20240116.2")
set(ABSEIL_URL     "https://github.com/abseil/abseil-cpp/archive/refs/tags/${ABSEIL_VERSION}.tar.gz")
set(ABSEIL_HASH    "SHA256=733726b8c3a6d39a4120d7e45ea8b41a434cdacde401cba500f14236c49b39dc")

set(NEON_FFT_VERSION "v0.1.0")
set(NEON_FFT_URL     "https://github.com/Ryuk17/neon-fft/archive/refs/tags/${NEON_FFT_VERSION}.tar.gz")
set(NEON_FFT_HASH    "SHA256=4b557d080a09c9eb5d1852b32daa8f022816772c58a9addc19c032f5b89a30d4")

set(PFFFT_VERSION "0aec0327a6912e1a0ec5326eef737c2ce19bc836")  # jpommier/pffft @ master
set(PFFFT_URL     "https://bitbucket.org/jpommier/pffft/get/${PFFFT_VERSION}.tar.gz")
set(PFFFT_HASH    "SHA256=29b3b0bf83ace5e32ad4fb26a99fe13284240a4d4f1b762970aa19c3821ddb1f")

set(EIGEN_VERSION "3.4.0")
set(EIGEN_URL     "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz")
set(EIGEN_HASH    "SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")

if(WITH_MNN)
    include(cmake/mnn.cmake)

    # Generic MNN inference engine (mnn_engine). It is internal to the
    # algorithms built on top of it, so its sources are compiled directly into
    # the consuming target instead of producing a standalone library.
    set(AE_MNN_ENGINE_SRC
        "${AE_ROOT}/mnn_engine/mnn_engine.cc"
        "${AE_ROOT}/mnn_engine/mnn_model_reader.cc"
    )
endif()

# absl: the super build provides absl::strings; fetch it for standalone builds.
if(NOT TARGET absl::strings)
    set(ABSEIL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(abseil-cpp
        URL      ${ABSEIL_URL}
        URL_HASH ${ABSEIL_HASH}
    )
    FetchContent_MakeAvailable(abseil-cpp)   # provides absl::strings
endif()

# absl 20240116's cctz takes the WinRT time-zone lookup path on MinGW once
# NTDDI_WIN10_NI is defined (mingw-w64 SDK >= 10.0.22621), but those WinRT
# headers are incompatible with the cctz code (IReference redefinitions and
# renamed entry points). Keep MinGW on the registry/TZ fallback path — the
# same path older MinGW toolchains used.
if(MINGW)
    set(CCTZ_LOOKUP "${abseil-cpp_SOURCE_DIR}/absl/time/internal/cctz/src/time_zone_lookup.cc")
    if(EXISTS "${CCTZ_LOOKUP}")
        file(READ "${CCTZ_LOOKUP}" _cctz_src)
        string(REPLACE
            "(defined(NTDDI_WIN10_NI) && NTDDI_VERSION >= NTDDI_WIN10_NI))"
            "(defined(NTDDI_WIN10_NI) && NTDDI_VERSION >= NTDDI_WIN10_NI && !defined(__MINGW32__)))"
            _cctz_src "${_cctz_src}")
        file(WRITE "${CCTZ_LOOKUP}" "${_cctz_src}")
    endif()
endif()

# neon-fft / pffft / eigen are compiled straight into the libAE_* libraries,
# so we only need their sources, not their own CMake targets.
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

if(NOT DEFINED NEON_FFT_SOURCE_DIR)
    FetchContent_Declare(neon-fft
        URL      ${NEON_FFT_URL}
        URL_HASH ${NEON_FFT_HASH}
    )
    FetchContent_GetProperties(neon-fft)
    if(NOT neon-fft_POPULATED)
        FetchContent_Populate(neon-fft)
    endif()
    set(NEON_FFT_SOURCE_DIR "${neon-fft_SOURCE_DIR}")
endif()

if(NOT DEFINED PFFFT_SOURCE_DIR)
    FetchContent_Declare(pffft
        URL      ${PFFFT_URL}
        URL_HASH ${PFFFT_HASH}
    )
    FetchContent_GetProperties(pffft)
    if(NOT pffft_POPULATED)
        FetchContent_Populate(pffft)
    endif()
    set(PFFFT_SOURCE_DIR "${pffft_SOURCE_DIR}")
endif()

if(BUILD_AE_DR AND NOT DEFINED EIGEN3_ROOT)
    FetchContent_Declare(eigen
        URL      ${EIGEN_URL}
        URL_HASH ${EIGEN_HASH}
    )
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()
    set(EIGEN3_ROOT "${eigen_SOURCE_DIR}" CACHE PATH "Eigen source root" FORCE)
endif()

# Vendored sources include "third_party/pffft/pffft.h" (WebRTC layout); bridge
# that path to wherever PFFFT_SOURCE_DIR actually lives (submodule or FetchContent).
set(PFFFT_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/pffft_compat/third_party/pffft")
file(MAKE_DIRECTORY "${PFFFT_COMPAT_DIR}/src")
file(WRITE "${PFFFT_COMPAT_DIR}/pffft.h"
     "#include \"${PFFFT_SOURCE_DIR}/pffft.h\"\n")
file(WRITE "${PFFFT_COMPAT_DIR}/src/pffft.h"
     "#include \"${PFFFT_SOURCE_DIR}/pffft.h\"\n")

set(CMAKE_VERBOSE_MAKEFILE ON)

# Disable WebRTC metrics (no-op stubs)
add_definitions(-DRTC_DISABLE_METRICS)

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

# ============================================================================
# libAE_VAD + test_ae_vad
# ============================================================================
if(BUILD_AE_VAD)
    add_library(AE_VAD STATIC
        "${AE_ROOT}/interface/audio_engine_vad.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        "${AE_DSP}/randomization_functions.c"
        "${AE_DSP}/cross_correlation.c"
        "${AE_CORE}/voice_activity_detection/vad_core/webrtc_vad.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_core.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_filterbank.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_gmm.c"
        "${AE_CORE}/voice_activity_detection/vad_core/vad_sp.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
    )

    file(GLOB VAD_SRC "${AE_CORE}/voice_activity_detection/*.cc")
    # silero_vad.cc depends on MNN; it is added explicitly below when
    # WITH_MNN=ON so the default DSP-only build stays MNN-free.
    list(FILTER VAD_SRC EXCLUDE REGEX "silero_vad\\.cc$")
    file(GLOB ISAC_VAD_SRC "${AE_CORE}/voice_activity_detection/isac/main/source/*.c")
    target_sources(AE_VAD PRIVATE ${VAD_SRC} ${ISAC_VAD_SRC} ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_VAD PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_VAD PUBLIC absl::strings)
    target_compile_definitions(AE_VAD PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_VAD PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    if(WITH_MNN)
        # Silero VAD runs through the internal MNN engine; both are compiled
        # straight into AE_VAD (no standalone libraries).
        target_sources(AE_VAD PRIVATE
            "${AE_CORE}/voice_activity_detection/silero_vad.cc"
            ${AE_MNN_ENGINE_SRC}
        )
        target_include_directories(AE_VAD PRIVATE "${AE_ROOT}/mnn_engine")
        target_link_libraries(AE_VAD PRIVATE soundproof_mnn)
        target_compile_definitions(AE_VAD PRIVATE AE_WITH_MNN=1)
    endif()

    add_ae_test(test_ae_vad "test_ae_vad.cc" AE_VAD)
    if(WITH_MNN)
        target_compile_definitions(test_ae_vad PRIVATE AE_WITH_MNN=1)
    endif()
endif()

# ============================================================================
# libAE_KWS + test_ae_kws — zipformer2 transducer keyword spotting.
# Requires MNN (the model is executed through the internal mnn_engine).
# ============================================================================
if(BUILD_AE_KWS)
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_KWS=ON but WITH_MNN=OFF; skipping libAE_KWS")
    else()
        include(cmake/knf.cmake)

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
endif()

# ============================================================================
# libAE_ASR + test_ae_asr — streaming ASR with pluggable model backends
# (transducer: zipformer/zipformer2/conformer/lstm; paraformer).
# Requires MNN.
# ============================================================================
if(BUILD_AE_ASR)
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_ASR=ON but WITH_MNN=OFF; skipping libAE_ASR")
    else()
        include(cmake/knf.cmake)
        include(cmake/ssentencepiece.cmake)

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
endif()

# ============================================================================
# libAE_SR + test_ae_sr — speaker recognition (identification/verification)
# with pluggable embedding backends (wespeaker/3d-speaker/nemo).
# Requires MNN.
# ============================================================================
if(BUILD_AE_SR)
    if(NOT WITH_MNN)
        message(STATUS "BUILD_AE_SR=ON but WITH_MNN=OFF; skipping libAE_SR")
    else()
        include(cmake/knf.cmake)

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
endif()

# ============================================================================
# libAE_SRC + test_ae_src
# ============================================================================
if(BUILD_AE_SRC)
    add_library(AE_SRC STATIC
        "${AE_ROOT}/interface/audio_engine_src.cpp"
        ${UTILS_WAV}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
    )

    target_sources(AE_SRC PRIVATE ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_SRC PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
    )
    target_link_libraries(AE_SRC PUBLIC absl::strings)
    target_compile_definitions(AE_SRC PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_SRC PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_src "test_ae_src.cc" AE_SRC)
endif()

# ============================================================================
# libAE_NS + test_ae_ns
# ============================================================================
if(BUILD_AE_NS)
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
endif()

# ============================================================================
# libAE_AGC + test_ae_agc
# ============================================================================
if(BUILD_AE_AGC)
    add_library(AE_AGC STATIC
        "${AE_ROOT}/interface/audio_engine_agc.cpp"
        ${UTILS_WAV}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
    )

    file(GLOB AGC_SRC "${AE_CORE}/automatic_gain_control/*.cc")
    target_sources(AE_AGC PRIVATE ${AGC_SRC})

    target_include_directories(AE_AGC PUBLIC ${LIB_COMMON_INCLUDES})
    target_link_libraries(AE_AGC PUBLIC absl::strings)
    target_compile_definitions(AE_AGC PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AGC PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_agc "test_ae_agc.cc" AE_AGC)
endif()

# ============================================================================
# libAE_AGC2 + test_ae_agc2
# ============================================================================
if(BUILD_AE_AGC2)
    add_library(AE_AGC2 STATIC
        "${AE_ROOT}/interface/audio_engine_agc2.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        "${RNN_VAD_DIR}/rnn_vad_weights.cc"
        "${AE_DSP}/pffft_wrapper.cc"
        "${PFFFT_SOURCE_DIR}/pffft.c"
        "${PFFFT_SOURCE_DIR}/fftpack.c"
    )

    file(GLOB AGC2_CORE "${AE_CORE}/automatic_gain_control2/*.cc")
    file(GLOB RNN_VAD_SRC "${AE_CORE}/automatic_gain_control2/rnn_vad/*.cc")
    target_sources(AE_AGC2 PRIVATE ${AGC2_CORE} ${RNN_VAD_SRC} ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/automatic_gain_control2/rnn_vad/vector_math_avx2.cc"
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_AGC2 PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
        "${RNN_VAD_DIR}"
        "${PFFFT_SOURCE_DIR}/"
        "${CMAKE_CURRENT_BINARY_DIR}/pffft_compat"
    )
    target_link_libraries(AE_AGC2 PUBLIC absl::strings)
    target_compile_definitions(AE_AGC2 PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AGC2 PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_agc2 "test_ae_agc2.cc" AE_AGC2)
endif()

# ============================================================================
# libAE_AECM + test_ae_aecm
# ============================================================================
if(BUILD_AE_AECM)
    add_library(AE_AECM STATIC
        "${AE_ROOT}/interface/audio_engine_aecm.cpp"
        ${UTILS_WAV}
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        "${AE_DSP}/downsample_fast.c"
        "${AE_DSP}/randomization_functions.c"
        "${AE_DSP}/cross_correlation.c"
        "${AE_ROOT}/utils/ring_buffer.c"
        "${AE_DSP}/delay_estimator.cc"
        "${AE_DSP}/delay_estimator_wrapper.cc"
    )

    file(GLOB AECM_SRC "${AE_CORE}/acoustic_echo_cancellation_mobile/*.cc")
    list(FILTER AECM_SRC EXCLUDE REGEX ".*_mips\\.cc$")
    list(FILTER AECM_SRC EXCLUDE REGEX ".*_neon\\.cc$")
    target_sources(AE_AECM PRIVATE ${AECM_SRC})

    target_include_directories(AE_AECM PUBLIC
        ${LIB_COMMON_INCLUDES}
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_AECM PUBLIC absl::strings)
    target_compile_definitions(AE_AECM PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AECM PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_aecm "test_ae_aecm.cc" AE_AECM)
endif()

# ============================================================================
# libAE_AEC + test_ae_aec
# ============================================================================
if(BUILD_AE_AEC)
    add_library(AE_AEC STATIC
        "${AE_ROOT}/interface/audio_engine_aec.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/echo_canceller3_config.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        "${AE_DSP}/high_pass_filter.cc"
        "${AE_DSP}/cascaded_biquad_filter.cc"
    )

    file(GLOB AEC3_SRC "${AE_CORE}/acoustic_echo_cancellation3/*.cc")
    list(FILTER AEC3_SRC EXCLUDE REGEX ".*neural_feature_extractor\\.cc$")
    list(FILTER AEC3_SRC EXCLUDE REGEX ".*neural_residual_echo_estimator_impl\\.cc$")
    target_sources(AE_AEC PRIVATE ${AEC3_SRC} ${RESAMPLER_SRC} ${RESAMPLER_DEPS})

    set_source_files_properties(
        "${AE_CORE}/acoustic_echo_cancellation3/adaptive_fir_filter_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/adaptive_fir_filter_erl_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/fft_data_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/matched_filter_avx2.cc"
        "${AE_CORE}/acoustic_echo_cancellation3/vector_math_avx2.cc"
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_AEC PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_AEC PUBLIC absl::strings)
    target_compile_definitions(AE_AEC PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_AEC PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_aec "test_ae_aec.cc" AE_AEC)
endif()

# ============================================================================
# libAE_BF + test_ae_bf
# ============================================================================
if(BUILD_AE_BF)
    add_library(AE_BF STATIC
        "${AE_ROOT}/interface/audio_engine_bf.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        "${AE_DSP}/randomization_functions.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        # BF core sources
        "${AE_CORE}/beamforming/nonlinear_beamformer.cc"
        "${AE_CORE}/beamforming/array_util.cc"
        "${AE_CORE}/beamforming/covariance_matrix_generator.cc"
        # BF dependencies (common_audio)
        "${AE_DSP}/lapped_transform.cc"
        "${AE_DSP}/window_generator.cc"
        "${AE_ROOT}/utils/blocker.cc"
        "${AE_ROOT}/utils/ring_buffer.c"
        "${AE_ROOT}/utils/audio_ring_buffer.cc"
        # RealFourier (NE10-based R2C/C2R FFT)
        "${AE_DSP}/real_fourier.cc"
        "${AE_ROOT}/utils/aligned_malloc.cc"
    )

    target_sources(AE_BF PRIVATE ${RESAMPLER_SRC})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_BF PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/beamforming/"
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_BF PUBLIC absl::strings)
    target_compile_definitions(AE_BF PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_BF PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_bf "test_ae_bf.cc" AE_BF)
endif()

# ============================================================================
# libAE_HS + test_ae_hs
# ============================================================================
if(BUILD_AE_HS)
    add_library(AE_HS STATIC
        "${AE_ROOT}/interface/audio_engine_hs.cpp"
        ${UTILS_WAV}
        "${AE_CORE}/howling_suppressor/howling_core.c"
        "${AE_CORE}/howling_suppressor/biquad.c"
        "${AE_CORE}/howling_suppressor/tone_detect.c"
    )

    target_include_directories(AE_HS PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/howling_suppressor/"
    )
    target_link_libraries(AE_HS PUBLIC absl::strings)
    target_compile_definitions(AE_HS PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_HS PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_hs "test_ae_hs.cc" AE_HS)
endif()

# ============================================================================
# libAE_IE + test_ae_ie
# ============================================================================
if(BUILD_AE_IE)
    add_library(AE_IE STATIC
        "${AE_ROOT}/interface/audio_engine_ie.cpp"
        ${UTILS_WAV}
        "${AE_ROOT}/utils/string_builder.cc"
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        ${DSP_RESAMPLE}
        "${AE_DSP}/splitting_filter.c"
        "${AE_DSP}/randomization_functions.c"
        ${UTILS_AUDIO_BUFFER_AND_SPLITTING}
        # IE core sources
        "${AE_CORE}/intelligibility_enhancer/intelligibility_enhancer.cc"
        "${AE_CORE}/intelligibility_enhancer/intelligibility_utils.cc"
        # Shared deps (same as BF: lapped_transform, real_fourier, window_generator, blocker)
        "${AE_DSP}/lapped_transform.cc"
        "${AE_DSP}/window_generator.cc"
        "${AE_ROOT}/utils/blocker.cc"
        "${AE_ROOT}/utils/ring_buffer.c"
        "${AE_ROOT}/utils/audio_ring_buffer.cc"
        # RealFourier (NE10-based R2C/C2R FFT)
        "${AE_DSP}/real_fourier.cc"
        "${AE_ROOT}/utils/aligned_malloc.cc"
    )

    target_sources(AE_IE PRIVATE ${RESAMPLER_SRC})

    set_source_files_properties(
        "${AE_CORE}/sample_rate_conversion/sinc_resampler_avx2.cc"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )

    target_include_directories(AE_IE PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/intelligibility_enhancer/"
        "${AE_CORE}/sample_rate_conversion/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_IE PUBLIC absl::strings)
    target_compile_definitions(AE_IE PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_IE PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_ie "test_ae_ie.cc" AE_IE)
endif()

# ============================================================================
# libAE_TS + test_ae_ts
# ============================================================================
if(BUILD_AE_TS)
    add_library(AE_TS STATIC
        "${AE_ROOT}/interface/audio_engine_ts.cpp"
        ${UTILS_WAV}
        ${NE10_FFT_SRC}
        ${DSP_COMMON}
        "${AE_DSP}/randomization_functions.c"
        # TS core sources
        "${AE_CORE}/transient_suppressor/transient_suppressor.cc"
        "${AE_CORE}/transient_suppressor/transient_detector.cc"
        "${AE_CORE}/transient_suppressor/moving_moments.cc"
        "${AE_CORE}/transient_suppressor/wpd_node.cc"
        "${AE_CORE}/transient_suppressor/wpd_tree.cc"
        # FIR filter (plain C implementation)
        "${AE_ROOT}/utils/fir_filter.cc"
    )

    target_include_directories(AE_TS PUBLIC
        ${LIB_COMMON_INCLUDES}
        "${AE_CORE}/transient_suppressor/"
        "${AE_CORE}/noise_suppressor/"
        ${NE10_FFT_INCLUDE}
    )
    target_link_libraries(AE_TS PUBLIC absl::strings)
    target_compile_definitions(AE_TS PRIVATE AUDIO_ENGINE_EXPORTS)
    set_target_properties(AE_TS PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${AE_PROJECT_ROOT}/lib")

    add_ae_test(test_ae_ts "test_ae_ts.cc" AE_TS)
endif()

# ============================================================================
# libAE_DR + test_ae_dr
# ============================================================================
# Eigen (header-only, double-precision FFT required by the dereverberation
# EKF — kept on purpose, see real_fft.h). Fetched above when not provided;
# override with -DEIGEN3_ROOT=<eigen source root>.
if(BUILD_AE_DR)
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
endif()

# ============================================================================
# test_mnn_engine — only when WITH_MNN=ON. The engine sources are compiled
# straight into the test executable (no standalone library).
# ============================================================================
if(WITH_MNN)
    add_executable(test_mnn_engine
        "${AE_TESTS}/test_mnn_engine.cc"
        ${AE_MNN_ENGINE_SRC}
    )

    target_include_directories(test_mnn_engine PRIVATE
        ${LIB_COMMON_INCLUDES}
        "${AE_ROOT}/mnn_engine"
    )

    target_link_libraries(test_mnn_engine PRIVATE soundproof_mnn)
    if(WIN32 AND MINGW)
        target_link_libraries(test_mnn_engine PRIVATE winmm)
    endif()

    add_test(NAME test_mnn_engine COMMAND test_mnn_engine
             WORKING_DIRECTORY "${AE_PROJECT_ROOT}")
endif()
