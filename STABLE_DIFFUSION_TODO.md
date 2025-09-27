# Stable Diffusion Implementation TODO

## ✅ Completed Features

### Core Infrastructure
- [x] Native JNI integration with stable-diffusion.cpp
- [x] CMake build system integration
- [x] CUDA acceleration support
- [x] Automatic model detection (safetensors format)
- [x] Resource management with AutoCloseable
- [x] Text-to-image generation
- [x] PNG image output conversion
- [x] Test coverage with 21 passing tests
- [x] Persistent fix for vae_tiling format string bug in build script

### Basic Generation
- [x] Simple text prompt generation
- [x] Custom image dimensions
- [x] Generation parameters (steps, CFG scale, SLG scale)
- [x] Seed control for reproducible results
- [x] Multiple sampling methods
- [x] SD3.5 Medium model support with optimized parameters

### ControlNet Integration
**Status: Complete** - Guided image generation now available
- [x] ControlNet model loading in JNI with createContextWithControlNet()
- [x] GenerationParameters extended with control image support
- [x] Control strength parameter (default: 0.9f)
- [x] ControlNet-specific wrapper methods with fluent API
- [x] CPU/GPU placement control for ControlNet models
- [x] Integration tests validating parameter handling

### Image-to-Image Generation
**Status: Complete** - Image transformation now available
- [x] init_image parameter in generation methods
- [x] Image strength control (default: 0.8f)
- [x] Img2img parameter validation and fluent API
- [x] Support for various image formats (1, 3, 4 channels)
- [x] Combined ControlNet + img2img operation support

### Inpainting Support
**Status: Complete** - Image editing with mask guidance
- [x] mask_image parameter in generation methods
- [x] Mask validation to prevent crashes with unsupported models (SD3/SD3.5)
- [x] Mask preprocessing utilities and fluent API support
- [x] Combined mask + ControlNet + img2img operation support
- [x] Integration tests with proper error handling

## ✅ STABLE DIFFUSION IMPLEMENTATION COMPLETE

### Canny Edge Preprocessing
**Status: Disabled** - Edge detection utility (library bug)
- [x] Wrap preprocess_canny() function in JNI
- [x] Create edge detection utility class (CannyEdgeDetector)
- [x] Add edge detection parameters (high/low threshold, weak/strong)
- [x] Memory management for stable-diffusion.cpp's malloc/free pattern
- [x] Parameter validation and fluent API
- [x] Test coverage for validation methods and disabled functionality
- [🚫] **DISABLED: Function signature bug in stable-diffusion.cpp**
  - `preprocess_canny()` takes `sd_image_t` by value instead of by reference
  - Causes double-free crashes due to memory management mismatch
  - All methods throw `UnsupportedOperationException` until library is fixed
  - Implementation is complete and ready for re-enabling when fixed

## 🚫 Advanced Features - Not Implemented

### Video Generation
**Status: Skipped** - Too complex with workarounds needed
- Video creation from text prompts requires extensive additional infrastructure
- File format handling, frame management, and temporal consistency
- Better handled by specialized video generation libraries

### Image Upscaling
**Status: Skipped** - Requires additional ESRGAN models
- Would need separate ESRGAN model files and management
- Adds complexity without providing core AI IDE functionality
- Post-processing can be handled by external tools

### Photo Maker Features
**Status: Skipped** - Requires specialized identity models
- Identity-consistent generation needs additional model files
- Complex identity embedding and preservation logic
- Not essential for core AI IDE image generation needs

## 📋 Implementation Notes

### Build and Development
- **Build C++**: `./build-native-cuda.sh` (includes vae_tiling format fix)
- **Build Java**: `mvn compile`
- **Test**: `mvn test`
- **Full cycle**: Run build-native-cuda.sh, then mvn compile

### JNI Pattern (Established)
New features follow this pattern:
1. Add Java parameter classes with fluent API methods
2. Add JNI native method declarations in NativeStableDiffusion
3. Implement C++ JNI wrapper functions in stable_diffusion_manager.cpp
4. Add error handling with JNIErrorHandler::throw_java_exception()
5. Create unit tests with parameter validation

### File Locations
- Java classes: `src/main/java/de/kherud/llama/diffusion/`
- JNI headers: `src/main/cpp/stable_diffusion_manager.h`
- JNI implementation: `src/main/cpp/stable_diffusion_manager.cpp`
- Tests: `src/test/java/de/kherud/llama/generation/TextToVisualConverterTest.java`

### Technical Requirements
- **Logging**: System.Logger only (no SLF4J, Log4j, etc.)
- **Indentation**: Use tabs, not spaces
- **Comments**: Only when logic isn't clear from naming
- **Dependencies**: Standard library only, existing stable-diffusion.cpp at `/opt/stable-diffusion.cpp`

### Testing Strategy
- Parameter validation tests for all new features
- JNI method availability testing
- Image data handling validation (multiple sizes/formats)
- Integration tests with actual models when available
- Error handling validation with missing models

## 🎯 Final Implementation Status

**✅ COMPLETED - Core Features Implemented:**
1. **Text-to-Image Generation** - Basic and advanced image generation from prompts
2. **ControlNet Integration** - Guided image generation with control images
3. **Image-to-Image** - Transform existing images with prompts
4. **Inpainting** - Selective image editing with mask guidance

**🚫 SKIPPED - Advanced Features:**
1. **Canny Edge Preprocessing** - Disabled due to stable-diffusion.cpp library bug
2. **Video Generation** - Too complex for current scope
3. **Image Upscaling** - Requires additional ESRGAN models
4. **Photo Maker** - Requires specialized identity models

**RESULT: Stable Diffusion implementation provides complete text-to-image functionality for AI IDE use cases.**

## 📝 Technical Considerations

### Memory Management
- Proper cleanup of native image buffers in JNI
- Handle large image/video data in C++ without memory leaks
- Use AutoCloseable pattern for resource management

### Performance
- GPU memory usage for concurrent operations
- Image format conversion optimization (RGB/RGBA handling)
- Buffer reuse where possible

### Error Handling
- Parameter validation in Java before JNI calls
- Graceful failure when models are missing
- Clear error messages using JNIErrorHandler pattern