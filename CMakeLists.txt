cmake_minimum_required(VERSION 3.15)

project(soundproof VERSION 1.0)

option(BUILD_PCM_IO "whether to build PCM IO library" ON)
option(BUILD_TESTING "whether to build unit test" ON)


if(BUILD_PCM_IO)
	set(SUPPORT_PCM TRUE)
endif()

if(BUILD_TESTING)
	add_subdirectory(tests)
endif()

add_subdirectory(src)



