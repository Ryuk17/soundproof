# simple-sentencepiece — BPE encoder used by the ASR punctuation model.
#
# Same pinned version / checksum as sherpa-mnn. Only the core static library
# (ssentencepiece_core) is built.

set(SSENTENCEPIECE_VERSION "0.7")
set(SSENTENCEPIECE_URL "https://github.com/pkufool/simple-sentencepiece/archive/refs/tags/v${SSENTENCEPIECE_VERSION}.tar.gz")
set(SSENTENCEPIECE_HASH "SHA256=1748a822060a35baa9f6609f84efc8eb54dc0e74b9ece3d82367b7119fdc75af")

if(NOT TARGET ssentencepiece_core)
    set(SBPE_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SBPE_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

    # Allow using a pre-downloaded tarball when GitHub is unreachable.
    set(_ssp_url ${SSENTENCEPIECE_URL})
    foreach(_ssp_candidate
            "${CMAKE_BINARY_DIR}/simple-sentencepiece-${SSENTENCEPIECE_VERSION}.tar.gz"
            "$ENV{HOME}/Downloads/simple-sentencepiece-${SSENTENCEPIECE_VERSION}.tar.gz")
        if(EXISTS "${_ssp_candidate}")
            set(_ssp_url "${_ssp_candidate}")
            message(STATUS "Using local simple-sentencepiece: ${_ssp_url}")
            break()
        endif()
    endforeach()

    FetchContent_Declare(simple-sentencepiece
        URL      ${_ssp_url}
        URL_HASH ${SSENTENCEPIECE_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )

    # Older cmake_minimum_required declarations are rejected by CMake 4.x.
    set(_sbpe_saved_policy_min "${CMAKE_POLICY_VERSION_MINIMUM}")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    # The subproject declares option(BUILD_SHARED_LIBS ON); force a static
    # library so libAE_ASR has no runtime DLL dependency.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    # Upstream checks an MSVC-only flag when WIN32 is set, which fails under
    # MinGW. Predefine the result so the check is skipped.
    if(WIN32 AND NOT MSVC)
        set(SBPE_COMPILER_SUPPORTS_CXX14 1 CACHE INTERNAL "" FORCE)
    endif()

    FetchContent_GetProperties(simple-sentencepiece)
    if(NOT simple-sentencepiece_POPULATED)
        FetchContent_Populate(simple-sentencepiece)
    endif()

    # Upstream guards MSVC-only "/wd..." flags with if(WIN32), which breaks
    # MinGW. Patch the source (idempotent) before adding the subproject.
    if(MINGW)
        set(_sbpe_cmake "${simple-sentencepiece_SOURCE_DIR}/CMakeLists.txt")
        if(EXISTS "${_sbpe_cmake}")
            file(READ "${_sbpe_cmake}" _sbpe_content)
            string(REPLACE
                "if(WIN32)\n  # disable various warnings for MSVC"
                "if(MSVC)\n  # disable various warnings for MSVC"
                _sbpe_patched "${_sbpe_content}")
            if(NOT _sbpe_patched STREQUAL _sbpe_content)
                file(WRITE "${_sbpe_cmake}" "${_sbpe_patched}")
            endif()
        endif()
    endif()

    add_subdirectory(${simple-sentencepiece_SOURCE_DIR}
                     ${simple-sentencepiece_BINARY_DIR})

    if(_sbpe_saved_policy_min)
        set(CMAKE_POLICY_VERSION_MINIMUM "${_sbpe_saved_policy_min}")
    else()
        unset(CMAKE_POLICY_VERSION_MINIMUM)
    endif()

    # PUBLIC: the target needs this include root for its own sources too.
    target_include_directories(ssentencepiece_core PUBLIC
        ${simple-sentencepiece_SOURCE_DIR}/)
endif()

message(STATUS "simple-sentencepiece ${SSENTENCEPIECE_VERSION} ready")
