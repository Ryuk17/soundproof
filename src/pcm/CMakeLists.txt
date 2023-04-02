add_library(sp_pcm STATIC
  sp_pcm_io.c
  sp_pcm_core.c
)

target_link_libraries(sp_pcm PUBLIC sp_common)