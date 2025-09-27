# Java Utility Classes Refactoring TODO

## Overview
This document outlines the refactoring plan for 22 Java utility classes with main methods in the java-llama.cpp project. These classes were primarily CLI tools translated from Python equivalents in `/opt/llama.cpp` and need to be transformed into library-friendly components.

## Refactoring Priority Classification

### High Priority (Core Library Components)

#### 1. **GGUFInspector**
- **File**: `src/main/java/de/kherud/llama/gguf/GGUFInspector.java`
- **Purpose**: GGUF file inspection and metadata analysis
- **Current Args**: `<gguf_file> [--no-metadata] [--no-tensors] [--verbose] [--json] [--filter key]`
- **Library Refactoring Needed**:
  - Builder pattern for options configuration
  - Fluent API for inspection operations
  - Stream API for metadata/tensor iteration
  - Separate validation from inspection operations
- **Impact**: High - Core functionality for GGUF file analysis

#### 2. **HuggingFaceToGGUFConverter**
- **File**: `src/main/java/de/kherud/llama/converters/HuggingFaceToGGUFConverter.java`
- **Purpose**: Convert HuggingFace models to GGUF format
- **Current Args**: `<model_path> <output_path> [--quantize TYPE] [--threads N] [--verbose]`
- **Library Refactoring Needed**:
  - Builder pattern with source/destination configuration
  - Asynchronous conversion with progress callbacks
  - Validation methods separate from conversion
  - Event-driven progress reporting
- **Impact**: High - Essential for model format conversion

#### 3. **ModelValidator**
- **File**: `src/main/java/de/kherud/llama/validation/ModelValidator.java`
- **Purpose**: Model accuracy and integrity validation
- **Current Args**: `<model_files> [--compare model2] [--nmse-tolerance N] [--no-checksums] [--threads N]`
- **Library Refactoring Needed**:
  - Builder pattern for validation configuration
  - Separate validation components (checksum, accuracy, consistency)
  - Batch validation support
  - Asynchronous validation operations
- **Impact**: High - Critical for model quality assurance

#### 4. **LlamaDevelopmentUtils**
- **File**: `src/main/java/de/kherud/llama/tools/LlamaDevelopmentUtils.java`
- **Purpose**: Development utilities and debugging tools
- **Current Args**: `sysinfo`, `monitor [--duration N]`, `memory [--interactive]`, `threads`, `threaddump`
- **Library Refactoring Needed**:
  - Separate utility classes for each function
  - Event-driven monitoring with callbacks
  - Builder pattern for monitoring configuration
  - Structured result objects instead of console output
- **Impact**: High - Essential for development and debugging

### Medium Priority (Valuable Library Components)

#### 5. **GGUFHasher**
- **File**: `src/main/java/de/kherud/llama/gguf/GGUFHasher.java`
- **Purpose**: Multi-algorithm file hashing for GGUF files
- **Current Args**: `<files> [-a ALGORITHMS] [-r] [-o output] [-t threads]`
- **Library Refactoring Needed**:
  - Builder pattern for hash configuration
  - Async hashing with progress callbacks
  - Batch processing support
  - Verification methods separate from hashing
- **Impact**: Medium - Important for file integrity

#### 6. **HuggingFaceDownloader**
- **File**: `src/main/java/de/kherud/llama/huggingface/HuggingFaceDownloader.java`
- **Purpose**: Download models from HuggingFace Hub
- **Current Args**: `<command> [download|info|search|list-cache] <args> [--token TOKEN] [--output DIR]`
- **Library Refactoring Needed**:
  - Builder pattern for download configuration
  - Async downloads with progress tracking
  - Cache management API
  - Model search and info methods
- **Impact**: Medium - Useful for model acquisition

#### 7. **GGUFMetadataEditor**
- **File**: `src/main/java/de/kherud/llama/gguf/GGUFMetadataEditor.java`
- **Purpose**: Edit GGUF file metadata
- **Current Args**: `<gguf_file> [--set key value] [--delete key] [--rename old new] [--json operations.json]`
- **Library Refactoring Needed**:
  - Transaction-like operations for metadata editing
  - Builder pattern for edit operations
  - Batch operation support from JSON
  - Validation before editing
- **Impact**: Medium - Useful for metadata manipulation

#### 8. **LoRAToGGUFConverter**
- **File**: `src/main/java/de/kherud/llama/converters/LoRAToGGUFConverter.java`
- **Purpose**: Convert LoRA adapters to GGUF format
- **Current Args**: `<adapter_path> <output_path> [--arch NAME] [--merge-norms] [--verbose]`
- **Library Refactoring Needed**:
  - Builder pattern for conversion options
  - Validation methods for LoRA structure
  - Async conversion with progress callbacks
  - Metadata extraction methods
- **Impact**: Medium - Specialized but important for LoRA workflows

#### 9. **HuggingFaceModelConverter**
- **File**: `src/main/java/de/kherud/llama/huggingface/HuggingFaceModelConverter.java`
- **Purpose**: Alternative model conversion utility
- **Library Refactoring Needed**: Similar to HuggingFaceToGGUFConverter
- **Impact**: Medium - May be redundant with #2

#### 10. **ProjectBuilder**
- **File**: `src/main/java/de/kherud/llama/tools/ProjectBuilder.java`
- **Purpose**: Build automation and project management
- **Current Args**: `<command> [build|clean|test] [--target TYPE] [--profile PROFILE] [--no-tests] [--verbose]`
- **Library Refactoring Needed**:
  - Builder pattern for build configuration
  - Async build operations
  - Project discovery and analysis methods
  - Incremental build support
- **Impact**: Medium - Useful for development workflows

### Lower Priority (Testing, Benchmarking, and Examples)

#### Testing Utilities
- **TokenizerTester** (`src/main/java/de/kherud/llama/testing/TokenizerTester.java`)
- **TokenizerBenchmark** (`src/main/java/de/kherud/llama/testing/TokenizerBenchmark.java`)
- **TokenizerComparator** (`src/main/java/de/kherud/llama/testing/TokenizerComparator.java`)
- **ServerTestFramework** (`src/main/java/de/kherud/llama/testing/ServerTestFramework.java`)
- **ServerEmbeddingHttpClient** (`src/main/java/de/kherud/llama/testing/ServerEmbeddingHttpClient.java`)
- **UtilityFunctionsTest** (`src/main/java/de/kherud/llama/test/UtilityFunctionsTest.java`)

#### Benchmark Utilities
- **ServerBenchmark** (`src/main/java/de/kherud/llama/benchmark/ServerBenchmark.java`)

#### Example Classes
- **SamplingExample** (`src/main/java/de/kherud/llama/examples/SamplingExample.java`)
- **LoRATrainingExample** (`src/main/java/de/kherud/llama/training/LoRATrainingExample.java`)

#### Other Utilities
- **VisionLanguageModel** (`src/main/java/de/kherud/llama/multimodal/VisionLanguageModel.java`)
- **ImageProcessor** (`src/main/java/de/kherud/llama/multimodal/ImageProcessor.java`)
- **OSInfo** (`src/main/java/de/kherud/llama/OSInfo.java`)

## Common Refactoring Patterns to Apply

### 1. Builder Pattern Implementation
```java
public static class Builder {
    public Builder option1(Type value) { ... }
    public Builder option2(Type value) { ... }
    public TargetClass build() { ... }
}
```

### 2. Asynchronous Operations
```java
public CompletableFuture<Result> operationAsync() { ... }
public void operationWithProgress(Consumer<Progress> callback) { ... }
```

### 3. Resource Management
```java
public class UtilityClass implements AutoCloseable {
    @Override
    public void close() { ... }
}
```

### 4. Validation Separation
```java
public boolean validate() { ... }
public ValidationResult validateDetailed() { ... }
public Result process() throws ValidationException { ... }
```

### 5. Batch Operations
```java
public List<Result> processBatch(List<Input> inputs) { ... }
public CompletableFuture<List<Result>> processBatchAsync(List<Input> inputs) { ... }
```

### 6. Event-Driven Design
```java
public void operation(Consumer<ProgressEvent> progressCallback) { ... }
public Stream<Event> streamEvents() { ... }
```

## Implementation Tasks

### Phase 1: High Priority Classes (Q1)
- [ ] Refactor GGUFInspector to library component
- [ ] Refactor HuggingFaceToGGUFConverter to library component
- [ ] Refactor ModelValidator to library component
- [ ] Refactor LlamaDevelopmentUtils to library component
- [ ] Create comprehensive unit tests for refactored classes
- [ ] Update documentation and examples

### Phase 2: Medium Priority Classes (Q2)
- [ ] Refactor GGUFHasher to library component
- [ ] Refactor HuggingFaceDownloader to library component
- [ ] Refactor GGUFMetadataEditor to library component
- [ ] Refactor LoRAToGGUFConverter to library component
- [ ] Evaluate and potentially merge HuggingFaceModelConverter with HuggingFaceToGGUFConverter
- [ ] Refactor ProjectBuilder to library component

### Phase 3: Lower Priority Classes (Q3)
- [ ] Evaluate testing utilities for potential library patterns
- [ ] Update example classes to use refactored library components
- [ ] Consider creating a testing framework from individual test utilities
- [ ] Update benchmark utilities to use library components
- [ ] Refactor remaining utilities as needed

### Phase 4: Integration and Cleanup (Q4)
- [ ] Create unified API documentation
- [ ] Implement integration tests
- [ ] Performance testing and optimization
- [ ] Create migration guide from CLI to library usage
- [ ] Clean up deprecated CLI interfaces (if appropriate)

## Success Criteria

### Library Component Qualities
- ✅ **Composability**: Can be easily combined with other components
- ✅ **Testability**: Individual methods can be unit tested
- ✅ **Reusability**: Core functionality can be reused in different contexts
- ✅ **Flexibility**: Builder pattern allows for different configurations
- ✅ **Performance**: Asynchronous operations don't block calling threads
- ✅ **Observability**: Progress callbacks and events provide insight
- ✅ **Error Handling**: Structured exception handling and validation
- ✅ **Integration**: Easy to integrate into larger applications

### Backward Compatibility
- Maintain CLI interfaces during transition period
- Provide clear migration path for existing users
- Document breaking changes and alternatives

## Notes
- Classes marked with "*test*", "*benchmark*", or "*example*" are deprioritized as they are primarily for validation and demonstration rather than core functionality
- Focus on classes that provide essential functionality for model processing, validation, and conversion
- Consider creating a unified API that ties these components together
- Maintain the existing CLI functionality while adding library APIs