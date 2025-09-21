# LoRA Training

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that allows adapting large language models with minimal computational overhead.

## Overview

LoRA decomposes weight updates into low-rank matrices, dramatically reducing the number of trainable parameters while maintaining adaptation quality.

### Key Concepts

- **Low-Rank Decomposition**: `ΔW = B × A` where `rank(B × A) << rank(W)`
- **Parameter Efficiency**: Train only ~0.1% of original parameters
- **Adaptation Quality**: Maintains performance comparable to full fine-tuning
- **Modularity**: Adapters can be loaded/unloaded dynamically

## Implementation Architecture

```
Original Weight: W [d_out, d_in]
LoRA Adaptation: ΔW = α * B * A
  - Matrix A: [r, d_in]  (random init)
  - Matrix B: [d_out, r] (zero init)
  - α: scaling factor
  - r: rank (typically 4-64)

Final Weight: W' = W + ΔW
```

## Core Components

### [Forward Pass](forward-pass.md)
Mathematical implementation of `output_delta = α * B * A * input`

### [Backward Pass](backward-pass.md)
Gradient computation for training the LoRA matrices

### [Training Pipeline](training-pipeline.md)
Complete training workflow from data loading to adapter saving

### [Matrix Initialization](matrix-initialization.md)
Strategies for initializing LoRA matrices for optimal training

## Configuration

### LoRA Parameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|--------|
| `rank` | Low-rank dimension | 4-64 | Higher = more expressive, more parameters |
| `alpha` | Scaling factor | 8-32 | Higher = stronger adaptation |
| `dropout` | Regularization rate | 0.1-0.3 | Prevents overfitting |
| `targetModules` | Which layers to adapt | `["q_proj", "v_proj"]` | Determines adaptation scope |

### Training Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `learningRate` | Optimizer step size | 1e-5 to 1e-3 |
| `batchSize` | Examples per batch | 1-16 |
| `epochs` | Training iterations | 1-10 |
| `weightDecay` | L2 regularization | 0.01-0.1 |

## Usage Examples

### Basic Training

```java
// Configure LoRA
LoRAConfig loraConfig = LoRAConfig.builder()
    .rank(8)
    .alpha(16.0f)
    .targetModules("q_proj", "v_proj")
    .dropout(0.1f)
    .build();

// Configure training
TrainingConfig trainingConfig = TrainingConfig.builder()
    .epochs(3)
    .batchSize(4)
    .learningRate(1e-4f)
    .outputDir("./adapters")
    .build();

// Train
LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
trainer.train(dataset);
```

### Advanced Configuration

```java
// High-rank configuration for complex tasks
LoRAConfig advancedConfig = LoRAConfig.builder()
    .rank(64)
    .alpha(32.0f)
    .targetModules("q_proj", "k_proj", "v_proj", "o_proj")
    .dropout(0.2f)
    .maxSequenceLength(2048)
    .gradientCheckpointing(true)
    .build();
```

## Output Compatibility

The training system generates GGUF-format adapter files that are fully compatible with llama.cpp's native LoRA loading:

```java
// Save adapter
trainer.saveLoRAAdapter("adapter.gguf");

// Load in llama.cpp
long handle = model.loadLoRAAdapter("adapter.gguf");
model.setLoRAAdapter(handle, 1.0f);
```

## Performance Characteristics

### Memory Usage
- **Base Model**: Unchanged (read-only)
- **LoRA Matrices**: ~0.1% of base model size
- **Training**: Additional optimizer state (Adam momentum)

### Training Speed
- **Gradient Computation**: Only for LoRA parameters
- **Memory Bandwidth**: Reduced due to smaller parameter set
- **Convergence**: Typically faster than full fine-tuning

### Quality Metrics
- **Parameter Efficiency**: 99%+ reduction in trainable parameters
- **Adaptation Quality**: 90-95% of full fine-tuning performance
- **Generalization**: Better than full fine-tuning on small datasets

## Best Practices

### Parameter Selection
1. **Start Small**: Begin with rank=4-8, increase if needed
2. **Alpha Tuning**: Typically `alpha = 2 * rank`
3. **Module Selection**: Start with attention projections (`q_proj`, `v_proj`)

### Training Strategy
1. **Learning Rate**: Lower than full fine-tuning (1e-4 to 1e-5)
2. **Batch Size**: Smaller batches often work better
3. **Regularization**: Use dropout and weight decay

### Debugging
1. **Loss Monitoring**: Should decrease steadily
2. **Gradient Norms**: Check for gradient explosion/vanishing
3. **Validation**: Monitor overfitting on held-out data

## Mathematical Background

### Low-Rank Decomposition
Given a weight matrix `W ∈ ℝ^(d×k)`, LoRA approximates updates as:
```
ΔW = BA^T
where B ∈ ℝ^(d×r), A ∈ ℝ^(k×r), and r << min(d,k)
```

### Parameter Count
```
Original parameters: d × k
LoRA parameters: d × r + k × r = r × (d + k)
Reduction ratio: r × (d + k) / (d × k)
```

For `d = k = 4096` and `r = 8`:
```
Original: 16,777,216 parameters
LoRA: 65,536 parameters (0.39% of original)
```

## Research References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)