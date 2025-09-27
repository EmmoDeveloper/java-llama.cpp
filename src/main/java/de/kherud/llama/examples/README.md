# VisionLanguageModelLibrary and ImageProcessorLibrary Examples

This directory contains comprehensive examples and implementations for the VisionLanguageModelLibrary and ImageProcessorLibrary, showcasing multimodal AI capabilities for image processing and vision-language inference.

## Files Overview

### Core Libraries
- **VisionLanguageModelLibrary** (`src/main/java/de/kherud/llama/multimodal/VisionLanguageModelLibrary.java`)
  - Multimodal AI library for vision-language tasks
  - Image captioning, visual question answering, image analysis, OCR
  - Builder pattern with configurable parameters
  - Async operations and batch processing

- **ImageProcessorLibrary** (`src/main/java/de/kherud/llama/multimodal/ImageProcessorLibrary.java`)
  - Image preprocessing for vision models
  - Resize, crop, normalize with various interpolation methods
  - Feature extraction and validation
  - Batch processing with progress tracking

### Example Implementations

#### 1. VisionLanguageExample.java
Demonstrates VisionLanguageModelLibrary usage with real-world scenarios:
- **Basic Image Captioning**: Generate descriptions for images
- **Visual Question Answering**: Answer questions about image content
- **Detailed Image Analysis**: Comprehensive scene analysis
- **OCR Functionality**: Extract text from images
- **Image Comparison**: Compare and contrast multiple images
- **Batch Processing**: Process multiple image-text pairs efficiently
- **Async Operations**: Non-blocking inference operations
- **Model Validation**: Verify model capabilities

```java
// Basic usage example
try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
        .languageModel("models/llava-v1.5-7b-q4_k_m.gguf")
        .maxTokens(100)
        .temperature(0.7f)
        .build()) {

    InferenceResult result = vlm.captionImage(Paths.get("image.jpg"));
    System.out.println("Caption: " + result.getResponse());
}
```

#### 2. ImageProcessorExample.java
Comprehensive image processing demonstrations:
- **Basic Processing**: Default image preprocessing
- **Configured Processing**: Custom normalization and interpolation
- **Batch Processing**: Multiple images with progress tracking
- **Async Processing**: Non-blocking image operations
- **Feature Extraction**: Extract pixel arrays for ML models
- **Validation**: Image format and integrity checking
- **Resizing**: Various resize operations
- **Normalization**: ImageNet vs range normalization
- **Performance Testing**: Benchmark different interpolation methods

```java
// Basic usage example
try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
        .targetSize(224, 224)
        .maintainAspectRatio(true)
        .meanNormalization(new float[]{0.485f, 0.456f, 0.406f})
        .build()) {

    ProcessingResult result = processor.processImage(Paths.get("image.jpg"));
    float[] features = result.getProcessedImage().get().flatten();
}
```

#### 3. MultimodalPipelineExample.java
Integrated pipeline examples showing real-world usage:
- **Complete Analysis Pipeline**: End-to-end image analysis with preprocessing
- **Batch Processing Pipeline**: High-throughput image processing
- **AI IDE Integration**: Code analysis, UI mockups, diagrams
- **Performance Optimization**: Parallel processing with thread pools
- **Error Handling**: Robust error recovery and validation

```java
// Pipeline example
try (ImageProcessorLibrary imageProcessor = ImageProcessorLibrary.builder()
        .targetSize(224, 224)
        .build();
     VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
        .languageModel(modelPath)
        .build()) {

    // 1. Validate and process image
    ValidationResult validation = imageProcessor.validateImage(imagePath);
    ProcessingResult processing = imageProcessor.processImage(imagePath);

    // 2. Generate AI analysis
    InferenceResult caption = vlm.captionImage(imagePath);
    InferenceResult analysis = vlm.analyzeImage(imagePath);
}
```

### Test Implementation

#### VisionLanguageLibrarySimpleTest.java
JUnit 4 test suite covering:
- Builder pattern validation
- Image processing functionality
- Batch operations
- Feature extraction
- Normalization methods
- Interpolation algorithms
- Error handling
- Resource management
- Integration testing

## Key Features

### VisionLanguageModelLibrary Features
- **Multiple Inference Types**: Caption, VQA, Analysis, OCR, Comparison
- **Configurable Parameters**: Temperature, top-p, top-k, max tokens
- **Progress Callbacks**: Real-time processing updates
- **Async Support**: CompletableFuture-based operations
- **Batch Processing**: Efficient multi-image inference
- **Vision Configuration**: Embedding dimensions, projection settings
- **Error Handling**: Comprehensive error reporting and recovery

### ImageProcessorLibrary Features
- **Flexible Resizing**: Maintain aspect ratio with center cropping
- **Multiple Interpolation**: Bicubic, bilinear, nearest neighbor
- **Normalization Options**: ImageNet standard or custom range
- **Feature Formats**: CHW and HWC pixel ordering
- **Batch Operations**: Process multiple images efficiently
- **Validation**: Image format and integrity checking
- **Progress Tracking**: Real-time processing updates
- **Memory Efficiency**: Optimized pixel handling

## Usage Patterns

### For AI IDE Integration
1. **Code Screenshot Analysis**: Analyze programming language and functionality
2. **UI Mockup Processing**: Describe interface elements and layout
3. **Diagram Understanding**: Explain architecture and data flow
4. **Documentation OCR**: Extract text from technical documents

### For Production Applications
1. **Content Moderation**: Analyze image content for safety
2. **Product Cataloging**: Generate descriptions for e-commerce
3. **Accessibility**: Create alt-text for images
4. **Quality Assurance**: Automated image quality checking

### Performance Optimization
- Use batch processing for multiple images
- Configure appropriate batch sizes (8-32 images)
- Utilize async operations for non-blocking inference
- Employ thread pools for parallel processing
- Choose optimal interpolation methods for speed vs quality

## Configuration Examples

### High-Quality Processing
```java
ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
    .targetSize(512, 512)
    .maintainAspectRatio(true)
    .centerCrop(true)
    .interpolation(InterpolationMethod.BICUBIC)
    .meanNormalization(new float[]{0.485f, 0.456f, 0.406f})
    .stdNormalization(new float[]{0.229f, 0.224f, 0.225f})
    .build();
```

### Performance-Optimized Processing
```java
ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
    .targetSize(224, 224)
    .maintainAspectRatio(false)
    .interpolation(InterpolationMethod.BILINEAR)
    .batchSize(32)
    .build();
```

### Detailed VLM Configuration
```java
VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
    .languageModel("models/llava-v1.5-7b-q4_k_m.gguf")
    .contextSize(4096)
    .maxTokens(300)
    .temperature(0.3f)
    .topP(0.9f)
    .topK(40)
    .enableBatchProcessing(true)
    .batchSize(8)
    .build();
```

## Running the Examples

1. **Compile the project**:
   ```bash
   mvn compile
   ```

2. **Run specific examples**:
   ```bash
   java -cp target/classes de.kherud.llama.examples.VisionLanguageExample
   java -cp target/classes de.kherud.llama.examples.ImageProcessorExample
   java -cp target/classes de.kherud.llama.examples.MultimodalPipelineExample
   ```

3. **Run tests**:
   ```bash
   mvn test -Dtest=VisionLanguageLibrarySimpleTest
   ```

## Requirements

- Java 21+
- Maven 3.6+
- LLaVA or compatible vision-language model (for VLM examples)
- Sample images in `examples/images/` directory
- Sufficient memory for model loading (8GB+ recommended)

## Notes

- Examples marked with `@Disabled` require actual model files
- Test images are generated programmatically for testing
- All libraries implement `AutoCloseable` for proper resource management
- Progress callbacks are optional but recommended for long-running operations
- Error handling is comprehensive with detailed error messages