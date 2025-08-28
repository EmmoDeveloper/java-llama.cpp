#!/bin/bash
# Build script for creating CUDA-enabled native libraries

set -e

echo "Building Java-llama.cpp with CUDA support"
echo "========================================="

# Step 1: Compile Java classes to generate JNI headers
echo "Step 1: Compiling Java classes to generate JNI headers..."
mvn compile

# Step 2: Build native library with CUDA
echo "Step 2: Building native library with CUDA support..."
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

echo "Build completed successfully!"
