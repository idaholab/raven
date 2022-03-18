
set(headers
        cppfmu/cppfmu_cs.hpp
        cppfmu/cppfmu_common.hpp

        fmi/fmi2Functions.h
        fmi/fmi2FunctionTypes.h
        fmi/fmi2TypesPlatform.h

        pythonfmu/PySlaveInstance.hpp
        pythonfmu/PyState.hpp
        )

set(sources
        cppfmu/cppfmu_cs.cpp
        cppfmu/fmi_functions.cpp
        pythonfmu/PySlaveInstance.cpp
        )

add_library(pythonfmu-export ${sources} ${headers})
target_compile_features(pythonfmu-export PUBLIC "cxx_std_17")

target_include_directories(pythonfmu-export
        PRIVATE
        "${Python3_INCLUDE_DIRS}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        )

target_link_libraries(pythonfmu-export
        PRIVATE
        ${Python3_LIBRARIES}
        )

if (WIN32)
  set_target_properties(pythonfmu-export
          PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../resources/binaries/${TARGET_PLATFORM}"
          )
else ()
  set_target_properties(pythonfmu-export
          PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../resources/binaries/${TARGET_PLATFORM}"
          )
endif ()
