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
    include(${AE_PROJECT_ROOT}/cmake/mnn.cmake)

    # Generic MNN inference engine (mnn_engine). It is internal to the
    # algorithms built on top of it, so its sources are compiled directly into
    # the consuming target instead of producing a standalone library.
    set(AE_MNN_ENGINE_SRC
        "${AE_ROOT}/mnn_engine/mnn_engine.cc"
        "${AE_ROOT}/mnn_engine/mnn_model_reader.cc"
    )

    # MNN-LLM shared lifecycle wrapper (RAG / LLM only). Kept separate from
    # AE_MNN_ENGINE_SRC: it includes llm/llm.hpp and must not leak into
    # WITH_MNN builds that do not enable WITH_MNN_LLM.
    if(WITH_MNN_LLM)
        set(AE_MNN_LLM_SRC
            "${AE_ROOT}/mnn_engine/mnn_llm_engine.cc"
        )
    endif()
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
