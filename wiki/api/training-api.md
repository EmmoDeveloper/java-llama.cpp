# Training API Reference

Complete API documentation for the LoRA training system.

## LoRATrainer

The main class for training LoRA adapters.

### Constructor

```java
public LoRATrainer(LlamaModel baseModel, LoRAConfig loraConfig, TrainingConfig trainingConfig)
```

**Parameters:**
- `baseModel`: The base language model to adapt
- `loraConfig`: LoRA-specific configuration
- `trainingConfig`: Training process configuration

### Methods

#### train()

```java
public void train(List<TrainingApplication> dataset)
```

Trains the LoRA adapter on the provided dataset.

**Parameters:**
- `dataset`: List of training examples

**Throws:**
- `RuntimeException`: If training fails

#### saveLoRAAdapter()

```java
public void saveLoRAAdapter(String filepath)
```

Saves the trained LoRA adapter to a GGUF file.

**Parameters:**
- `filepath`: Output path for the adapter file

#### Getters

```java
public LoRAConfig getLoRAConfig()
public TrainingConfig getTrainingConfig()
public Map<String, LoRAModule> getLoRAModules()
```

## LoRAConfig

Configuration for LoRA adaptation parameters.

### Builder Pattern

```java
LoRAConfig config = LoRAConfig.builder()
    .rank(8)
    .alpha(16.0f)
    .dropout(0.1f)
    .targetModules("q_proj", "v_proj")
    .maxSequenceLength(2048)
    .gradientCheckpointing(true)
    .build();
```

### Parameters

#### rank
- **Type**: `int`
- **Default**: `16`
- **Range**: `1-512` (typically `4-64`)
- **Description**: Low-rank dimension controlling adapter expressiveness

#### alpha
- **Type**: `float`
- **Default**: `32.0f`
- **Range**: `1.0-128.0` (typically `2*rank`)
- **Description**: LoRA scaling factor

#### dropout
- **Type**: `float`
- **Default**: `0.1f`
- **Range**: `0.0-0.5`
- **Description**: Dropout rate for regularization

#### targetModules
- **Type**: `String[]`
- **Default**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Options**:
  - `"q_proj"`: Query projection
  - `"k_proj"`: Key projection
  - `"v_proj"`: Value projection
  - `"o_proj"`: Output projection
- **Description**: Which attention modules to adapt

#### maxSequenceLength
- **Type**: `int`
- **Default**: `2048`
- **Range**: `128-8192`
- **Description**: Maximum sequence length for training

#### gradientCheckpointing
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Trade computation for memory during training

### Example Configurations

#### Low-Resource Training
```java
LoRAConfig lowResource = LoRAConfig.builder()
    .rank(4)
    .alpha(8.0f)
    .dropout(0.0f)
    .targetModules("q_proj", "v_proj")
    .build();
```

#### High-Quality Adaptation
```java
LoRAConfig highQuality = LoRAConfig.builder()
    .rank(32)
    .alpha(64.0f)
    .dropout(0.2f)
    .targetModules("q_proj", "k_proj", "v_proj", "o_proj")
    .maxSequenceLength(4096)
    .build();
```

#### Code-Specific Training
```java
LoRAConfig codeTraining = LoRAConfig.builder()
    .rank(16)
    .alpha(32.0f)
    .dropout(0.1f)
    .targetModules("q_proj", "v_proj")
    .maxSequenceLength(2048)
    .build();
```

## TrainingConfig

Configuration for the training process.

### Builder Pattern

```java
TrainingConfig config = TrainingConfig.builder()
    .epochs(3)
    .batchSize(4)
    .learningRate(1e-4f)
    .weightDecay(0.01f)
    .warmupSteps(100)
    .saveSteps(500)
    .outputDir("./lora_output")
    .build();
```

### Parameters

#### epochs
- **Type**: `int`
- **Default**: `3`
- **Range**: `1-20`
- **Description**: Number of training epochs

#### batchSize
- **Type**: `int`
- **Default**: `4`
- **Range**: `1-32`
- **Description**: Number of examples per batch

#### learningRate
- **Type**: `float`
- **Default**: `2e-4f`
- **Range**: `1e-6 - 1e-2`
- **Description**: Adam optimizer learning rate

#### weightDecay
- **Type**: `float`
- **Default**: `0.01f`
- **Range**: `0.0-0.1`
- **Description**: L2 regularization coefficient

#### warmupSteps
- **Type**: `int`
- **Default**: `100`
- **Range**: `0-1000`
- **Description**: Learning rate warmup steps

#### saveSteps
- **Type**: `int`
- **Default**: `500`
- **Range**: `50-5000`
- **Description**: Steps between checkpoints

#### outputDir
- **Type**: `String`
- **Default**: `"./lora_output"`
- **Description**: Directory for saving adapters and checkpoints

### Example Configurations

#### Quick Prototyping
```java
TrainingConfig prototype = TrainingConfig.builder()
    .epochs(1)
    .batchSize(1)
    .learningRate(1e-3f)
    .saveSteps(100)
    .outputDir("./prototype")
    .build();
```

#### Production Training
```java
TrainingConfig production = TrainingConfig.builder()
    .epochs(5)
    .batchSize(8)
    .learningRate(5e-5f)
    .weightDecay(0.01f)
    .warmupSteps(500)
    .saveSteps(1000)
    .outputDir("./production_adapters")
    .build();
```

## TrainingApplication

Data structure representing a single training example.

### Record Definition

```java
public record TrainingApplication(String input, String target, String instruction, float weight)
```

### Factory Methods

#### instructionFormat()
```java
public static TrainingApplication instructionFormat(String instruction, String input, String response)
```

Creates an instruction-following training example.

**Example:**
```java
TrainingApplication example = TrainingApplication.instructionFormat(
    "Translate the following text to Spanish",
    "Hello, how are you?",
    "Hola, ¿cómo estás?"
);
```

#### chatFormat()
```java
public static TrainingApplication chatFormat(String systemPrompt, String userMessage, String assistantResponse)
```

Creates a chat-style training example.

**Example:**
```java
TrainingApplication example = TrainingApplication.chatFormat(
    "You are a helpful coding assistant",
    "Write a function to calculate factorial",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
);
```

#### completionFormat()
```java
public static TrainingApplication completionFormat(String prompt, String completion)
```

Creates a simple completion training example.

**Example:**
```java
TrainingApplication example = TrainingApplication.completionFormat(
    "def fibonacci(n):",
    "\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
);
```

### Methods

#### getFullText()
```java
public String getFullText()
```

Returns concatenated input and target text.

#### getFormattedForTraining()
```java
public String getFormattedForTraining()
```

Returns text with special training tokens.

### Weight Parameter

The `weight` parameter allows importance scaling:
- `1.0f`: Normal importance (default)
- `> 1.0f`: Higher importance
- `< 1.0f`: Lower importance
- `0.0f`: Skip this example

**Example:**
```java
// High-importance example
TrainingApplication important = new TrainingApplication(
    "Critical safety instruction",
    "Always follow safety protocols",
    null,
    2.0f  // 2x weight
);
```

## LoRAModule

Internal class representing a single LoRA adaptation module.

### Constructor

```java
public LoRAModule(String name, int inputDim, int outputDim, int rank)
```

### Methods

#### forward()
```java
public float[] forward(float[] input, float alpha, boolean training, float dropoutRate)
```

Computes the LoRA forward pass.

#### backward()
```java
public void backward(float[] inputActivation, float[] outputGrad, float alpha)
```

Accumulates gradients for training.

#### updateWeights()
```java
public void updateWeights(float learningRate, float beta1, float beta2, float epsilon, int step)
```

Updates weights using Adam optimizer.

### Getters

```java
public String getName()
public float[][] getMatrixA()
public float[][] getMatrixB()
public int getRank()
public int getInputDim()
public int getOutputDim()
```

## Error Handling

### Common Exceptions

#### RuntimeException
Thrown by `train()` method for various training failures:
- Out of memory
- Invalid configuration
- Model loading errors

#### IOException
Thrown by `saveLoRAAdapter()` for file system issues:
- Permission denied
- Disk full
- Invalid path

### Error Recovery

```java
try {
    trainer.train(dataset);
} catch (RuntimeException e) {
    if (e.getCause() instanceof OutOfMemoryError) {
        // Reduce batch size and retry
        trainingConfig = trainingConfig.toBuilder()
            .batchSize(trainingConfig.getBatchSize() / 2)
            .build();
        trainer = new LoRATrainer(model, loraConfig, trainingConfig);
        trainer.train(dataset);
    } else {
        throw e;
    }
}
```

## Performance Tuning

### Memory Optimization

```java
// Reduce memory usage
LoRAConfig memoryOptimized = LoRAConfig.builder()
    .rank(4)                    // Lower rank
    .targetModules("q_proj")    // Fewer modules
    .gradientCheckpointing(true) // Memory vs compute trade-off
    .build();

TrainingConfig batchOptimized = TrainingConfig.builder()
    .batchSize(1)              // Smaller batches
    .build();
```

### Speed Optimization

```java
// Faster training
TrainingConfig speedOptimized = TrainingConfig.builder()
    .batchSize(8)              // Larger batches
    .learningRate(1e-3f)       // Higher learning rate
    .epochs(1)                 // Fewer epochs
    .saveSteps(1000)           // Less frequent saving
    .build();
```

## Validation

### Configuration Validation

```java
public static void validateConfig(LoRAConfig config) {
    if (config.getRank() <= 0) {
        throw new IllegalArgumentException("Rank must be positive");
    }
    if (config.getAlpha() <= 0) {
        throw new IllegalArgumentException("Alpha must be positive");
    }
    if (config.getDropout() < 0 || config.getDropout() >= 1) {
        throw new IllegalArgumentException("Dropout must be in [0, 1)");
    }
    if (config.getTargetModules().length == 0) {
        throw new IllegalArgumentException("Must specify at least one target module");
    }
}
```

### Runtime Validation

```java
public static void validateTraining(LoRATrainer trainer, List<TrainingApplication> dataset) {
    if (dataset.isEmpty()) {
        throw new IllegalArgumentException("Dataset cannot be empty");
    }

    // Check for empty examples
    long emptyCount = dataset.stream()
        .filter(ex -> ex.input().trim().isEmpty() || ex.target().trim().isEmpty())
        .count();

    if (emptyCount > 0) {
        throw new IllegalArgumentException("Dataset contains " + emptyCount + " empty examples");
    }

    // Estimate memory requirements
    long estimatedMemory = trainer.getLoRAModules().size() *
                          trainer.getLoRAConfig().getRank() *
                          4096 * 4; // Approximate

    Runtime runtime = Runtime.getRuntime();
    if (estimatedMemory > runtime.maxMemory() * 0.8) {
        throw new IllegalArgumentException("Training may exceed available memory");
    }
}
```