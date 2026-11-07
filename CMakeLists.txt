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
#   Algorithm targets live in cmake/<algo>.cmake; this file only declares the
#   options and includes the files selected by those switches.
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
option(BUILD_AE_TTS  "Build libAE_TTS and test_ae_tts (requires WITH_MNN)" ON)
option(BUILD_AE_RAG "Build libAE_RAG and test_ae_rag (requires WITH_MNN_LLM)" ON)
option(BUILD_AE_TURN "Build libAE_TURN and test_ae_turn (requires WITH_MNN)" ON)
option(BUILD_AE_LLM  "Build libAE_LLM and test_ae_llm (requires WITH_MNN_VISION)" ON)

# Optional deep-learning runtime (MNN). OFF by default: heavy dependency
# (static lib ~12 MB, several minutes to build) used by future DNN modules.
option(WITH_MNN     "Build the MNN inference runtime"              OFF)
option(WITH_MNN_LLM "Build MNN with LLM support (implies WITH_MNN)" OFF)
option(WITH_MNN_VISION "Build MNN with vision support (implies WITH_MNN_LLM)" OFF)
if(WITH_MNN_VISION)
    set(WITH_MNN_LLM ON)
endif()
if(WITH_MNN_LLM)
    set(WITH_MNN ON)
endif()

# ============================================================================
# Dependencies, shared source groups and the add_ae_test helper
# ============================================================================
include(${AE_PROJECT_ROOT}/cmake/dependencies.cmake)
include(${AE_PROJECT_ROOT}/cmake/common_sources.cmake)

# ============================================================================
# Algorithm modules — cmake/<algo>.cmake, included according to the switches
# ============================================================================
if(BUILD_AE_VAD)
    include(${AE_PROJECT_ROOT}/cmake/vad.cmake)
endif()
if(BUILD_AE_KWS)
    include(${AE_PROJECT_ROOT}/cmake/kws.cmake)
endif()
if(BUILD_AE_ASR)
    include(${AE_PROJECT_ROOT}/cmake/asr.cmake)
endif()
if(BUILD_AE_SR)
    include(${AE_PROJECT_ROOT}/cmake/sr.cmake)
endif()
if(BUILD_AE_TTS)
    include(${AE_PROJECT_ROOT}/cmake/tts.cmake)
endif()
if(BUILD_AE_RAG)
    include(${AE_PROJECT_ROOT}/cmake/rag.cmake)
endif()
if(BUILD_AE_TURN)
    include(${AE_PROJECT_ROOT}/cmake/turn.cmake)
endif()
if(BUILD_AE_LLM)
    include(${AE_PROJECT_ROOT}/cmake/llm.cmake)
endif()
if(BUILD_AE_SRC)
    include(${AE_PROJECT_ROOT}/cmake/src.cmake)
endif()
if(BUILD_AE_NS)
    include(${AE_PROJECT_ROOT}/cmake/ns.cmake)
endif()
if(BUILD_AE_AGC)
    include(${AE_PROJECT_ROOT}/cmake/agc.cmake)
endif()
if(BUILD_AE_AGC2)
    include(${AE_PROJECT_ROOT}/cmake/agc2.cmake)
endif()
if(BUILD_AE_AECM)
    include(${AE_PROJECT_ROOT}/cmake/aecm.cmake)
endif()
if(BUILD_AE_AEC)
    include(${AE_PROJECT_ROOT}/cmake/aec.cmake)
endif()
if(BUILD_AE_BF)
    include(${AE_PROJECT_ROOT}/cmake/bf.cmake)
endif()
if(BUILD_AE_HS)
    include(${AE_PROJECT_ROOT}/cmake/hs.cmake)
endif()
if(BUILD_AE_IE)
    include(${AE_PROJECT_ROOT}/cmake/ie.cmake)
endif()
if(BUILD_AE_TS)
    include(${AE_PROJECT_ROOT}/cmake/ts.cmake)
endif()
if(BUILD_AE_DR)
    include(${AE_PROJECT_ROOT}/cmake/dr.cmake)
endif()

# Generic MNN engine smoke test (no standalone library).
if(WITH_MNN)
    include(${AE_PROJECT_ROOT}/cmake/test_mnn_engine.cmake)
endif()
