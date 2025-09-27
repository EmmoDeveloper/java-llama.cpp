cmake_minimum_required(VERSION 3.14)

project(jllama CXX)

#################### Use local llama.cpp source ####################

# Use the real llama.cpp source at /opt/llama.cpp
set(LLAMA_CPP_DIR "/opt/llama.cpp")

# Verify the source directory exists
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp source not found at ${LLAMA_CPP_DIR}")
endif()

# Force common library to be built
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common library" FORCE)

# Add llama.cpp as a subdirectory
add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)

#################### Use local stable-diffusion.cpp source ####################

# Use the real stable-diffusion.cpp source at /opt/stable-diffusion.cpp
set(STABLE_DIFFUSION_CPP_DIR "/opt/stable-diffusion.cpp")

# Check for separately built stable-diffusion
if(DEFINED STABLE_DIFFUSION_BUILD_DIR AND EXISTS "${STABLE_DIFFUSION_BUILD_DIR}/bin/libstable-diffusion.so")
    message(STATUS "Found separately built stable-diffusion at ${STABLE_DIFFUSION_BUILD_DIR}")
    set(BUILD_STABLE_DIFFUSION ON)
    set(STABLE_DIFFUSION_LIBRARY "${STABLE_DIFFUSION_BUILD_DIR}/bin/libstable-diffusion.so")
    set(STABLE_DIFFUSION_INCLUDE_DIR "${STABLE_DIFFUSION_CPP_DIR}")
elseif(NOT EXISTS "${STABLE_DIFFUSION_CPP_DIR}/CMakeLists.txt")
    message(STATUS "stable-diffusion.cpp not found at ${STABLE_DIFFUSION_CPP_DIR}")
    message(STATUS "Building without Stable Diffusion support")
    set(BUILD_STABLE_DIFFUSION OFF)
else()
    message(STATUS "Found stable-diffusion.cpp source but no built library")
    message(STATUS "Building without Stable Diffusion support")
    set(BUILD_STABLE_DIFFUSION OFF)
endif()

#################### jllama ####################

# Find OS and architecture info
if(NOT DEFINED OS_NAME)
    find_package(Java REQUIRED)
    find_program(JAVA_EXECUTABLE NAMES java)
    execute_process(
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --os
      OUTPUT_VARIABLE OS_NAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
if(NOT OS_NAME)
    message(FATAL_ERROR "Could not determine OS name")
endif()

if(NOT DEFINED OS_ARCH)
    find_package(Java REQUIRED)
    find_program(JAVA_EXECUTABLE NAMES java)
    execute_process(
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --arch
      OUTPUT_VARIABLE OS_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
if(NOT OS_ARCH)
    message(FATAL_ERROR "Could not determine CPU architecture")
endif()

set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources/de/kherud/llama/${OS_NAME}/${OS_ARCH})
message(STATUS "Installing files to ${JLLAMA_DIR}")

# Find JNI headers - configurable location
if(NOT DEFINED JNI_INCLUDE_DIRS)
    # Default to Java 17 OpenJDK location, but allow override
    set(JNI_INCLUDE_DIRS "/usr/lib/jvm/java-17-openjdk-amd64/include;/usr/lib/jvm/java-17-openjdk-amd64/include/linux" CACHE STRING "JNI include directories")
endif()

message(STATUS "Using JNI include directories: ${JNI_INCLUDE_DIRS}")
if(NOT JNI_INCLUDE_DIRS)
    message(FATAL_ERROR "Could not determine JNI include directories")
endif()

# Create the jllama shared library
add_library(jllama SHARED
    src/main/cpp/jllama.cpp
    src/main/cpp/jni_utils.cpp
    src/main/cpp/completion_task.cpp
    src/main/cpp/llama_server.cpp
    src/main/cpp/pattern_preprocessor.cpp
    src/main/cpp/memory_manager.cpp
    src/main/cpp/jni_logger.cpp
    src/main/cpp/model_manager.cpp
    src/main/cpp/tokenization_handler.cpp
    src/main/cpp/state_manager.cpp
    src/main/cpp/jni_error_handler.cpp
    src/main/cpp/lora_adapter_manager.cpp
    src/main/cpp/ai_sampler_manager.cpp
    src/main/cpp/kv_cache_manager.cpp
    src/main/cpp/model_info_manager.cpp
    src/main/cpp/quantization_manager.cpp
    src/main/cpp/embedding_manager.cpp
    src/main/cpp/completion_manager.cpp
    src/main/cpp/template_manager.cpp
    src/main/cpp/reranking_manager.cpp
    src/main/cpp/threading_manager.cpp
    src/main/cpp/schema_grammar_manager.cpp
    src/main/cpp/model_loader_manager.cpp
    src/main/cpp/utility_manager.cpp
    src/main/cpp/system_info_manager.cpp
    src/main/cpp/batch_manager.cpp
    src/main/cpp/training_manager.cpp
    # Add common sources needed for training
    ${LLAMA_CPP_DIR}/common/common.cpp
    ${LLAMA_CPP_DIR}/common/sampling.cpp
    ${LLAMA_CPP_DIR}/common/console.cpp
    ${LLAMA_CPP_DIR}/common/json-schema-to-grammar.cpp
    ${LLAMA_CPP_DIR}/common/log.cpp
)

# Conditionally add stable-diffusion manager
if(BUILD_STABLE_DIFFUSION)
    target_sources(jllama PRIVATE src/main/cpp/stable_diffusion_manager.cpp)
    target_compile_definitions(jllama PRIVATE BUILD_STABLE_DIFFUSION)
endif()

set_target_properties(jllama PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""  # Remove the "lib" prefix to get jllama.so instead of libjllama.so
)

target_include_directories(jllama PRIVATE
    src/main/cpp
    ${JNI_INCLUDE_DIRS}
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/vendor
)

# Conditionally add stable-diffusion include directories
if(BUILD_STABLE_DIFFUSION)
    target_include_directories(jllama PRIVATE ${STABLE_DIFFUSION_INCLUDE_DIR})
endif()

# Link against the real llama.cpp libraries
target_link_libraries(jllama PRIVATE
    llama
    ggml
    build_info
)

# Conditionally link stable-diffusion
if(BUILD_STABLE_DIFFUSION)
    target_link_libraries(jllama PRIVATE ${STABLE_DIFFUSION_LIBRARY})
endif()

target_compile_features(jllama PRIVATE cxx_std_17)

# Set output directories
if(OS_NAME STREQUAL "Windows")
    set_target_properties(jllama PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${JLLAMA_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${JLLAMA_DIR}
    )
else()
    set_target_properties(jllama PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${JLLAMA_DIR}
    )
endif()

# Copy metal file if needed
if (LLAMA_METAL AND NOT LLAMA_METAL_EMBED_LIBRARY)
    configure_file(${LLAMA_CPP_DIR}/ggml-metal.metal ${JLLAMA_DIR}/ggml-metal.metal COPYONLY)
endif()

# Copy stable-diffusion library if built
if(BUILD_STABLE_DIFFUSION)
    configure_file(${STABLE_DIFFUSION_LIBRARY} ${JLLAMA_DIR}/libstable-diffusion.so COPYONLY)
endif()

# Build training process executable
add_executable(training_process src/main/cpp/training_process.cpp)
target_include_directories(training_process PRIVATE
	${LLAMA_CPP_DIR}/include
	${LLAMA_CPP_DIR}/ggml/include
	${LLAMA_CPP_DIR}/common
	src/main/cpp  # For json.hpp
)
target_link_libraries(training_process PRIVATE llama common)
target_compile_features(training_process PRIVATE cxx_std_17)
set_target_properties(training_process PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${JLLAMA_DIR}
)