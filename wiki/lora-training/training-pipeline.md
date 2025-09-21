# LoRA Training Pipeline

The complete training pipeline orchestrates data processing, forward/backward passes, and optimization to create LoRA adapters.

## Pipeline Overview

```
Dataset → Preprocessing → Batching → Forward Pass → Loss Computation →
Backward Pass → Gradient Update → Checkpoint Saving → Adapter Export
```

## Training Loop Implementation

### Main Training Method

```java
public void train(List<TrainingApplication> dataset) {
    // Initialize metrics
    float bestLoss = Float.MAX_VALUE;
    long totalTrainingTime = 0;

    // Training loop
    for (int epoch = 0; epoch < trainingConfig.getEpochs(); epoch++) {
        long epochStartTime = System.currentTimeMillis();

        float epochLoss = trainEpoch(dataset, epoch);

        // Save checkpoints if loss improved
        if (epochLoss < bestLoss) {
            bestLoss = epochLoss;
            saveLoRAAdapter(String.format("%s/best_adapter_epoch_%d.gguf",
                          trainingConfig.getOutputDir(), epoch + 1));
        }

        // Periodic checkpointing
        if ((epoch + 1) % Math.max(1, trainingConfig.getEpochs() / 5) == 0) {
            saveLoRAAdapter(String.format("%s/checkpoint_epoch_%d.gguf",
                          trainingConfig.getOutputDir(), epoch + 1));
        }
    }

    // Final save
    saveLoRAAdapter(String.format("%s/final_adapter.gguf", trainingConfig.getOutputDir()));
}
```

### Epoch Training

```java
private float trainEpoch(List<TrainingApplication> dataset, int epoch) {
    // Shuffle data for each epoch
    List<TrainingApplication> mutableDataset = new ArrayList<>(dataset);
    Collections.shuffle(mutableDataset);

    float totalLoss = 0.0f;
    int numBatches = 0;

    // Process in batches
    for (int i = 0; i < mutableDataset.size(); i += trainingConfig.getBatchSize()) {
        List<TrainingApplication> batch = mutableDataset.subList(i,
            Math.min(i + trainingConfig.getBatchSize(), mutableDataset.size()));

        float batchLoss = trainBatch(batch);
        totalLoss += batchLoss;
        numBatches++;
        globalStep++;

        // Logging and checkpointing
        if (globalStep % 100 == 0) {
            LOGGER.info(String.format("Step %d: loss %.6f", globalStep, batchLoss));
        }

        if (globalStep % trainingConfig.getSaveSteps() == 0) {
            saveLoRAAdapter(String.format("%s/checkpoint-step-%d.gguf",
                          trainingConfig.getOutputDir(), globalStep));
        }
    }

    return totalLoss / numBatches;
}
```

## Batch Processing

### Single Batch Training

```java
private float trainBatch(List<TrainingApplication> batch) {
    float batchLoss = 0.0f;

    for (TrainingApplication example : batch) {
        // Forward pass with LoRA
        float loss = computeLossWithLoRA(example);
        batchLoss += loss;

        // Backward pass
        updateGradients(example, loss);
    }

    // Update weights using accumulated gradients
    updateLoRAWeights();

    return batchLoss / batch.size();
}
```

## Loss Computation

### Language Modeling Loss

```java
private float computeLossWithLoRA(TrainingApplication example) {
    String fullText = example.getFullText();
    int[] tokens = baseModel.encode(fullText);

    if (tokens.length < 2) {
        return 0.0f; // Cannot compute loss on single token
    }

    // Split into input and target portions
    String inputText = example.input();
    int[] inputTokens = baseModel.encode(inputText);
    int inputLen = inputTokens.length;

    float totalLoss = 0.0f;
    int lossCount = 0;

    // Compute cross-entropy loss for target portion
    for (int pos = inputLen; pos < tokens.length - 1; pos++) {
        float[] logits = computeLogitsAtPosition(tokens, pos);
        int targetToken = tokens[pos + 1];

        // Apply LoRA modifications
        logits = applyLoRAToLogits(logits, pos, true);

        // Cross-entropy loss
        float loss = computeCrossEntropyLoss(logits, targetToken);
        totalLoss += loss;
        lossCount++;
    }

    return lossCount > 0 ? totalLoss / lossCount : 0.0f;
}
```

### Cross-Entropy Implementation

```java
private float computeCrossEntropyLoss(float[] logits, int targetToken) {
    float[] probs = softmax(logits);
    float targetProb = probs[targetToken];
    return (float) -Math.log(Math.max(targetProb, 1e-8f)); // Avoid log(0)
}

private float[] softmax(float[] logits) {
    float[] probs = new float[logits.length];

    // Find max for numerical stability
    float maxLogit = Float.NEGATIVE_INFINITY;
    for (float logit : logits) {
        maxLogit = Math.max(maxLogit, logit);
    }

    // Compute exp(logit - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < logits.length; i++) {
        probs[i] = (float) Math.exp(logits[i] - maxLogit);
        sum += probs[i];
    }

    // Normalize
    for (int i = 0; i < probs.length; i++) {
        probs[i] /= sum;
    }

    return probs;
}
```

## Gradient Computation

### Gradient Update Process

```java
private void updateGradients(TrainingApplication example, float loss) {
    String fullText = example.getFullText();
    int[] tokens = baseModel.encode(fullText);

    if (tokens.length < 2) return;

    String inputText = example.input();
    int[] inputTokens = baseModel.encode(inputText);
    int inputLen = inputTokens.length;

    // Compute gradients for each position
    for (int pos = inputLen; pos < tokens.length - 1; pos++) {
        int targetToken = tokens[pos + 1];

        // Get base logits and apply LoRA
        float[] baseLogits = computeLogitsAtPosition(tokens, pos);
        float[] modifiedLogits = applyLoRAToLogits(baseLogits, pos, true);

        // Compute loss gradient (∂loss/∂logits)
        float[] logitsGrad = computeLogitsGradient(modifiedLogits, targetToken);

        // Backpropagate through LoRA modules
        backpropagateLoRAGradients(logitsGrad, pos);
    }
}
```

### Logits Gradient

```java
private float[] computeLogitsGradient(float[] logits, int targetToken) {
    float[] probs = softmax(logits);
    float[] grad = new float[logits.length];

    // For cross-entropy: ∂L/∂logit_i = prob_i - δ_i
    for (int i = 0; i < grad.length; i++) {
        if (i == targetToken) {
            grad[i] = probs[i] - 1.0f;
        } else {
            grad[i] = probs[i];
        }
    }

    return grad;
}
```

## Weight Updates

### Adam Optimizer

```java
private void updateLoRAWeights() {
    for (LoRAModule module : loraModules.values()) {
        module.updateWeights(
            trainingConfig.getLearningRate(),
            0.9f,  // beta1
            0.999f, // beta2
            1e-8f,  // epsilon
            globalStep
        );
    }
}
```

### LoRA Module Weight Update

```java
public void updateWeights(float learningRate, float beta1, float beta2, float epsilon, int step) {
    float beta1Power = (float) Math.pow(beta1, step);
    float beta2Power = (float) Math.pow(beta2, step);
    float correctedLR = learningRate * (float) Math.sqrt(1 - beta2Power) / (1 - beta1Power);

    // Update Matrix A
    for (int i = 0; i < rank; i++) {
        for (int j = 0; j < inputDim; j++) {
            momentumA1[i][j] = beta1 * momentumA1[i][j] + (1 - beta1) * gradA[i][j];
            momentumA2[i][j] = beta2 * momentumA2[i][j] + (1 - beta2) * gradA[i][j] * gradA[i][j];

            float update = correctedLR * momentumA1[i][j] / ((float) Math.sqrt(momentumA2[i][j]) + epsilon);
            matrixA[i][j] -= update;
            gradA[i][j] = 0; // Reset gradient
        }
    }

    // Update Matrix B (similar process)
    // ...
}
```

## Data Pipeline

### Training Data Formats

The pipeline supports multiple data formats through `TrainingApplication`:

```java
// Instruction-following format
TrainingApplication.instructionFormat(
    "Translate the following text",
    "Hello world",
    "Hola mundo"
);

// Chat format
TrainingApplication.chatFormat(
    "You are a helpful assistant",
    "What is the capital of France?",
    "The capital of France is Paris."
);

// Completion format
TrainingApplication.completionFormat(
    "def fibonacci(n):",
    "\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
);
```

### Dataset Processing

```java
// Load from various formats
List<TrainingApplication> dataset = DatasetProcessor.loadAlpacaDataset("alpaca.json");
dataset.addAll(DatasetProcessor.loadJsonlDataset("completions.jsonl"));

// Filter by length
dataset = DatasetProcessor.filterByLength(dataset, 2048);

// Train/validation split
Map<String, List<TrainingApplication>> split =
    DatasetProcessor.trainValidationSplit(dataset, 0.1f);
```

## Checkpointing Strategy

### Automatic Checkpointing

1. **Best Model**: Saved when validation loss improves
2. **Periodic**: Every N epochs (configurable)
3. **Step-based**: Every N training steps
4. **Final**: Always saved at training completion

### Checkpoint Format

```
output_dir/
├── best_adapter_epoch_2.gguf      # Best validation loss
├── checkpoint_epoch_1.gguf        # Periodic checkpoint
├── checkpoint_epoch_3.gguf        # Periodic checkpoint
├── checkpoint-step-500.gguf       # Step-based checkpoint
└── final_adapter.gguf             # Final model
```

## Monitoring and Logging

### Training Metrics

```java
// Training summary
LOGGER.info("=== Training Summary ===");
LOGGER.info(String.format("Total training time: %.2f seconds", totalTrainingTime / 1000.0f));
LOGGER.info(String.format("Final loss: %.6f", bestLoss));
LOGGER.info(String.format("Total steps: %d", globalStep));
LOGGER.info(String.format("Average time per step: %.2f ms", (float) totalTrainingTime / globalStep));
```

### Loss Tracking

- Per-step loss logging (every 100 steps)
- Per-epoch average loss
- Best validation loss tracking
- Training time metrics

## Memory Management

### Gradient Accumulation

```java
// Gradients are accumulated across batch items
for (TrainingApplication example : batch) {
    updateGradients(example, loss); // Accumulates gradients
}
updateLoRAWeights(); // Single weight update per batch
```

### Memory Optimization

- **Gradient Checkpointing**: Optional memory vs. computation trade-off
- **Dropout**: Applied only during training
- **Batch Processing**: Configurable batch sizes
- **Tensor Reuse**: Minimize memory allocations

## Error Handling

### Training Resilience

```java
try {
    trainer.train(dataset);
} catch (OutOfMemoryError e) {
    // Reduce batch size and retry
    trainingConfig = trainingConfig.toBuilder()
        .batchSize(trainingConfig.getBatchSize() / 2)
        .build();
    trainer = new LoRATrainer(model, loraConfig, trainingConfig);
    trainer.train(dataset);
}
```

### Validation Checks

- Dataset size validation
- Memory requirement estimation
- Configuration parameter validation
- GGUF file integrity verification

## Integration Points

### Model Loading

```java
// Load base model
LlamaModel model = new LlamaModel(modelParams);

// Create trainer
LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
```

### Adapter Usage

```java
// After training
trainer.saveLoRAAdapter("adapter.gguf");

// Load and use
long handle = model.loadLoRAAdapter("adapter.gguf");
model.setLoRAAdapter(handle, 1.0f);

// Generate with adapter
String result = model.generate("prompt", generateParams);

// Cleanup
model.removeLoRAAdapter(handle);
model.freeLoRAAdapter(handle);
```