#!/bin/bash
# Build script for creating CUDA-enabled native libraries

set -e

echo "Building Java-llama.cpp with CUDA support"
echo "========================================="

# Step 1: Compile Java classes to generate JNI headers
echo "Step 1: Compiling Java classes to generate JNI headers..."
mvn compile

# Step 2: Build llama.cpp first (to get GGML for stable-diffusion)
echo "Step 2: Building llama.cpp first to get GGML..."
cmake -B build -DGGML_CUDA=ON
cmake --build build --target ggml --config Release -j8

# Step 3: Set up stable-diffusion.cpp with GGML from llama.cpp
echo "Step 3: Setting up stable-diffusion.cpp with GGML..."
STABLE_DIFFUSION_BUILD_DIR="./stable-diffusion-build"
if [ ! -d "$STABLE_DIFFUSION_BUILD_DIR" ]; then
    mkdir -p "$STABLE_DIFFUSION_BUILD_DIR"
fi

# Copy stable-diffusion.cpp source and set up GGML
cp -r /opt/stable-diffusion.cpp/* "$STABLE_DIFFUSION_BUILD_DIR/"
cp -r /opt/llama.cpp/ggml "$STABLE_DIFFUSION_BUILD_DIR/"

# Apply local fix for vae_tiling format string bug
echo "Applying local fix for vae_tiling format string bug..."
sed -i 's/"vae_tiling: %s\\n"//' "$STABLE_DIFFUSION_BUILD_DIR/stable-diffusion.cpp"

cd "$STABLE_DIFFUSION_BUILD_DIR"
cmake . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSD_BUILD_SHARED_LIBS=ON \
    -DSD_BUILD_EXAMPLES=OFF \
    -DSD_CUDA=ON
cmake --build . --config Release -j8
cd ..

# Step 4: Complete building native library with stable-diffusion
echo "Step 4: Completing native library build with stable-diffusion..."
cmake -B build -DGGML_CUDA=ON -DSTABLE_DIFFUSION_BUILD_DIR="$(pwd)/$STABLE_DIFFUSION_BUILD_DIR"
cmake --build build --config Release

echo "Build completed successfully!"
