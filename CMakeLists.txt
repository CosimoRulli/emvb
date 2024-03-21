cmake_minimum_required(VERSION 3.14)

project(EMVB)

enable_testing()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

MESSAGE(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

include_directories(include)

# Set compiler options
if(NOT MSVC)
#  set(CMAKE_C_FLAGS "-O3 -ffast-math -march=native -mavx512f  -funroll-loops -fopenmp -g ${CMAKE_C_FLAGS}")
  set(CMAKE_C_FLAGS "-O3 -ffast-math -march=native -mavx512f -mavx512vl -mavx512bw -funroll-loops -fopenmp -g ${CMAKE_C_FLAGS}")


else()
  include("${CMAKE_CURRENT_LIST_DIR}/msvc_simd_isa.cmake")
  if(SUPPORT_MSVC_AVX512)
    set(MSVC_SIMD_FLAGS "/arch:AVX512")
  elseif(SUPPORT_MSVC_AVX2)
    set(MSVC_SIMD_FLAGS "/arch:AVX2")
  elseif(SUPPORT_MSVC_AVX)
    set(MSVC_SIMD_FLAGS "/arch:AVX")
  else()
    set(MSVC_SIMD_FLAGS "")
  endif()
  set(CMAKE_C_FLAGS "/O2 /fp:fast ${MSVC_SIMD_FLAGS} ${CMAKE_C_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -Wall -Wextra -Wno-missing-braces")

# Set Windows stack size as on Linux: 2MB on 32-bit, 8MB on 64-bit
if (WIN32)
  math(EXPR stack_size "${CMAKE_SIZEOF_VOID_P}*${CMAKE_SIZEOF_VOID_P}*128*1024")
  if (MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:${stack_size}")
  else()
    # compiling with clang + lld
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xlinker /stack:${stack_size}")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS}")

## Project-wide setup
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS YES)


## External projects 
Include(FetchContent)


find_package(MKL CONFIG REQUIRED)

add_subdirectory(external/faiss)
include_directories(${PROJECT_SOURCE_DIR}/external/faiss)

add_subdirectory(external/cnpy)
include_directories(${PROJECT_SOURCE_DIR}/external/cnpy)

add_subdirectory(external/cmd_line_parser)
include_directories(${PROJECT_SOURCE_DIR}/external/cmd_line_parser)


add_executable(perf_emvb src/perf_emvb.cpp)
target_link_libraries(perf_emvb cnpy faiss $<LINK_ONLY:MKL::MKL>)
target_include_directories(perf_emvb PRIVATE include)
