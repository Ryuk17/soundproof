add_library(sp_common INTERFACE)
target_include_directories(sp_common INTERFACE ${PROJECT_SOURCE_DIR}/src/)

if(SUPPORT_PCM)
add_subdirectory(pcm)
endif()