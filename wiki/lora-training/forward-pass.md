# LoRA Forward Pass Implementation

The forward pass is the core mathematical operation in LoRA (Low-Rank Adaptation) training, implementing the formula:

```
output_delta = α * B * A * input
```

## Mathematical Foundation

LoRA decomposes weight updates into two low-rank matrices:
- **Matrix A**: `[rank, input_dim]` - Initialized with random Gaussian values
- **Matrix B**: `[output_dim, rank]` - Initialized to zero
- **α (alpha)**: Scaling factor controlling adaptation strength

The original weight matrix `W` is augmented as:
```
W_new = W_original + α * (B * A)
```

## Implementation Details

### Core Algorithm

```java
public float[] forward(float[] input, float alpha, boolean training, float dropoutRate) {
    // Apply input dropout during training
    float[] processedInput = input;
    if (training && dropoutRate > 0) {
        processedInput = applyDropout(input, dropoutRate);
    }

    // Step 1: A * input: [rank, input_dim] * [input_dim] = [rank]
    float[] aOutput = new float[rank];
    for (int i = 0; i < rank; i++) {
        for (int j = 0; j < inputDim; j++) {
            aOutput[i] += matrixA[i][j] * processedInput[j];
        }
    }

    // Step 2: B * aOutput: [output_dim, rank] * [rank] = [output_dim]
    float[] delta = new float[outputDim];
    for (int i = 0; i < outputDim; i++) {
        for (int j = 0; j < rank; j++) {
            delta[i] += matrixB[i][j] * aOutput[j];
        }
        delta[i] *= alpha; // Apply LoRA scaling
    }

    return delta;
}
```

### Step-by-Step Breakdown

#### 1. Input Preprocessing
```java
// Apply dropout during training for regularization
if (training && dropoutRate > 0) {
    processedInput = applyDropout(input, dropoutRate);
}
```

#### 2. First Matrix Multiplication (A * input)
```java
// Compute intermediate representation: [rank, input_dim] * [input_dim] = [rank]
float[] aOutput = new float[rank];
for (int i = 0; i < rank; i++) {
    for (int j = 0; j < inputDim; j++) {
        aOutput[i] += matrixA[i][j] * processedInput[j];
    }
}
```

#### 3. Second Matrix Multiplication (B * aOutput)
```java
// Project to output space: [output_dim, rank] * [rank] = [output_dim]
float[] delta = new float[outputDim];
for (int i = 0; i < outputDim; i++) {
    for (int j = 0; j < rank; j++) {
        delta[i] += matrixB[i][j] * aOutput[j];
    }
    delta[i] *= alpha; // Apply scaling factor
}
```

## Dropout Implementation

Inverted dropout is used for proper scaling during training:

```java
private float[] applyDropout(float[] input, float dropoutRate) {
    float[] result = new float[input.length];
    float scale = 1.0f / (1.0f - dropoutRate); // Inverted dropout scaling

    for (int i = 0; i < input.length; i++) {
        if (ThreadLocalRandom.current().nextFloat() < dropoutRate) {
            result[i] = 0.0f; // Drop this element
        } else {
            result[i] = input[i] * scale; // Scale remaining elements
        }
    }
    return result;
}
```

## Key Design Decisions

### Matrix Initialization
- **Matrix A**: Random Gaussian initialization with `std = sqrt(1/rank)`
- **Matrix B**: Zero initialization to ensure LoRA starts as identity

### Computational Efficiency
- Sequential matrix multiplications avoid creating large intermediate matrices
- In-place scaling reduces memory allocations
- Dropout is applied only during training

### Parameter Sensitivity
- **Rank**: Higher rank = more expressiveness but more parameters
- **Alpha**: Higher alpha = stronger adaptation effect
- **Dropout**: Typical values 0.1-0.3 for regularization

## Integration with Training

The forward pass output (delta) is added to the base model's activations:

```java
float[] baseOutput = baseModel.forward(input);
float[] loraOutput = loraModule.forward(activations, alpha, training, dropout);

// Combine base and LoRA outputs
for (int i = 0; i < baseOutput.length; i++) {
    baseOutput[i] += loraOutput[i];
}
```

## Performance Considerations

- **Memory**: O(rank * (input_dim + output_dim)) for matrices
- **Computation**: O(rank * input_dim + rank * output_dim) per forward pass
- **Typical values**: rank=4-64, input_dim=output_dim=4096 for transformers

## Related Documentation

- [Training Pipeline](training-pipeline.md) - How forward pass fits into training
- [Backward Pass](backward-pass.md) - Gradient computation
- [LoRA Configuration](../api/training-api.md#loraconfig) - Parameter tuning