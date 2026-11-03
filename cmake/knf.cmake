# kaldi-native-fbank — fbank feature extractor used by the KWS algorithm.
#
# Same pinned version / checksum as sherpa-mnn so that the 80-dim fbank
# features fed to the zipformer2 KWS model are bit-for-bit compatible.
# Only the core library target (kaldi-native-fbank-core) is built.

set(KNF_VERSION "1.21.1")
set(KNF_URL "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v${KNF_VERSION}.tar.gz")
set(KNF_HASH "SHA256=37c1aa230b00fe062791d800d8fc50aa3de215918d3dce6440699e67275d859e")

if(NOT TARGET kaldi-native-fbank-core)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(kaldi-native-fbank
        URL      ${KNF_URL}
        URL_HASH ${KNF_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )

    # kaldi-native-fbank 1.21.1 declares cmake_minimum_required(VERSION 3.3),
    # which CMake 4.x rejects; allow it for this subproject only.
    set(_knf_saved_policy_min "${CMAKE_POLICY_VERSION_MINIMUM}")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    # Build knf as a static library so libAE_KWS has no runtime DLL dependency.
    set(_knf_saved_shared_libs "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(kaldi-native-fbank)
    if(_knf_saved_shared_libs)
        set(BUILD_SHARED_LIBS "${_knf_saved_shared_libs}")
    else()
        unset(BUILD_SHARED_LIBS)
    endif()

    if(_knf_saved_policy_min)
        set(CMAKE_POLICY_VERSION_MINIMUM "${_knf_saved_policy_min}")
    else()
        unset(CMAKE_POLICY_VERSION_MINIMUM)
    endif()

    # The upstream target does not export its own include root.
    target_include_directories(kaldi-native-fbank-core INTERFACE
        ${kaldi-native-fbank_SOURCE_DIR}/)
endif()

message(STATUS "kaldi-native-fbank ${KNF_VERSION} ready")
