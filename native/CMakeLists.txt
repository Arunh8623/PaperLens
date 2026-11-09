# Builds the optional C++ accelerator `paperlens._native` (pybind11). Normally driven by scripts/build_native.py.
cmake_minimum_required(VERSION 3.15)
project(paperlens_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_native src/paperlens_native.cpp)
if(MSVC)
  # /fp:precise + no fast-math: floating point results must equal the Python reference bit for bit
  target_compile_options(_native PRIVATE /W4 /fp:precise)
else()
  # -ffp-contract=off: never fuse a*b+c into one rounding (it would differ from Python's double arithmetic)
  target_compile_options(_native PRIVATE -O2 -Wall -Wextra -ffp-contract=off)
endif()
if(DEFINED PAPERLENS_OUT_DIR)
  foreach(cfg "" _RELEASE _DEBUG _RELWITHDEBINFO _MINSIZEREL)
    set_target_properties(_native PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY${cfg} "${PAPERLENS_OUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY${cfg} "${PAPERLENS_OUT_DIR}")
  endforeach()
endif()
