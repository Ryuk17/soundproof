cmake_minimum_required(VERSION 3.15)

project(Soundproof VERSION 1.0)

option(ENABLE_PCM_IO "Support PCM file read and write" ON)


add_subdirectory(src)
if(ENABLE_PCM_IO)
  add_subdirectory(src/pcm)
endif()

add_executable(test_pcm main.cc)

