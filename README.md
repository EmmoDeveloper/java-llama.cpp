# Java Llama.cpp Project

A Java wrapper for llama.cpp with native LoRA training capabilities.

## Project Structure

### C++ Implementation
- **Original llama.cpp**: `/opt/llama.cpp/` (git pull available)
- **JNI bindings**: `src/main/cpp/` (build with `./build-native-cuda.sh`)

### Java Implementation
- **Maven structure**: Standard Java project layout
- **Native LoRA training**: Pure Java implementation with GGUF compatibility

## Project Wiki

This project includes comprehensive documentation in the `wiki/` directory:

### Core Concepts
- [LoRA Training](wiki/lora-training/README.md) - Low-Rank Adaptation implementation
- [GGUF Format](wiki/gguf-format/README.md) - File format specification and implementation
- [Model Converters](wiki/converters/README.md) - HuggingFace and LoRA conversion utilities
- [Model Integration](wiki/model-integration/README.md) - llama.cpp integration details

### Implementation Details
- [Forward Pass](wiki/lora-training/forward-pass.md) - Mathematical implementation of LoRA forward computation
- [Training Pipeline](wiki/lora-training/training-pipeline.md) - Complete training workflow
- [GGUF Writer](wiki/gguf-format/writer-implementation.md) - Native Java GGUF file writer
- [Server Benchmark](wiki/benchmarking/server-benchmark.md) - Performance testing tools

### API Reference
- [Training API](wiki/api/training-api.md) - LoRATrainer and configuration classes
- [Dataset Processing](wiki/api/dataset-processing.md) - Data loading and preprocessing
- [Model Loading](wiki/api/model-loading.md) - Model and adapter management
- [Conversion API](wiki/api/conversion-api.md) - Model and adapter conversion tools

## Quick Start

```java
// Configure LoRA training
LoRATrainer.LoRAConfig loraConfig = LoRATrainer.LoRAConfig.builder()
    .rank(8)
    .alpha(16.0f)
    .targetModules("q_proj")
    .build();

// Configure training parameters
LoRATrainer.TrainingConfig trainingConfig = LoRATrainer.TrainingConfig.builder()
    .epochs(3)
    .batchSize(4)
    .learningRate(1e-4f)
    .outputDir("./lora_output")
    .build();

// Create trainer and train
LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
trainer.train(dataset);
```

## Architecture

This project provides a native Java implementation of LoRA training that generates GGUF-compatible adapter files, eliminating the need for external Python dependencies while maintaining full compatibility with llama.cpp's native LoRA loading system.
