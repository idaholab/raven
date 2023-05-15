cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER_NAMES gcc clang icc cc)
set(CMAKE_CXX_COMPILER_NAMES g++ clang++ icpc c++ cxx)

project(pythonfmu-export VERSION 0.2.0)

# ==============================================================================
# Build settings
# ==============================================================================

set(BUILD_SHARED_LIBS ON)

# ==============================================================================
# Global internal configuration
# ==============================================================================

if (MSVC)
  # https://stackoverflow.com/questions/14172856/compile-with-mt-instead-of-md-using-cmake
  set(CompilerFlags
          CMAKE_CXX_FLAGS
          CMAKE_CXX_FLAGS_DEBUG
          CMAKE_CXX_FLAGS_RELEASE
          CMAKE_C_FLAGS
          CMAKE_C_FLAGS_DEBUG
          CMAKE_C_FLAGS_RELEASE
          )
  foreach (CompilerFlag ${CompilerFlags})
    string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
  endforeach ()
endif ()

# Automatically export all symbols in Windows DLLs.
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ==============================================================================
# Dependencies
# ==============================================================================

# Force to use stable Python ABI https://docs.python.org/3/c-api/stable.html
add_compile_definitions(Py_LIMITED_API)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (WIN32)
  set(Python3_LIBRARIES ${Python3_LIBRARY_DIRS}/python3.lib)
endif ()

# ==============================================================================
# Sources
# ==============================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/headers)

set(sources
        cpp/cppfmu_cs.cpp
        cpp/fmi_functions.cpp
        cpp/PySlaveInstance.cpp
        )

add_library(pythonfmu-export ${sources})
target_compile_features(pythonfmu-export PUBLIC "cxx_std_17")
target_include_directories(pythonfmu-export
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "${Python3_INCLUDE_DIRS}"
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        )

if (WIN32)
  target_link_libraries(pythonfmu-export
        PRIVATE
        ${Python3_LIBRARIES}
        )
elseif (APPLE)
  target_link_libraries(pythonfmu-export
        PRIVATE
        Python3::Module
        )
else ()
  target_link_libraries(pythonfmu-export
        PRIVATE
        Python3::Python
        )
endif ()

if (WIN32)
  set(TARGET_PLATFORM win)
elseif (APPLE)
  set(TARGET_PLATFORM darwin)
else ()
  set(TARGET_PLATFORM linux)
endif ()

if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
  set(TARGET_PLATFORM ${TARGET_PLATFORM}64)
else ()
  set(TARGET_PLATFORM ${TARGET_PLATFORM}32)
endif ()

message("Building pythonfmu-export for platform ${TARGET_PLATFORM}")

if (WIN32)
  set_target_properties(pythonfmu-export
          PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../resources/binaries/${TARGET_PLATFORM}"
          )
else ()
  set_target_properties(pythonfmu-export
          PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../resources/binaries/${TARGET_PLATFORM}"
          )
endif ()
