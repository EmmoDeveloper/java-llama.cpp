package de.kherud.llama.examples;

import de.kherud.llama.generation.TextToVisualConverter;
import de.kherud.llama.generation.TextToVisualConverterTypes.*;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Example demonstrating how to use TextToVisualConverter with Stable Diffusion v3.5 Medium models.
 *
 * This example shows how to:
 * 1. Configure the converter for your downloaded Stable Diffusion models
 * 2. Generate high-quality images from text prompts
 * 3. Use different quantization levels for different use cases
 */
public class StableDiffusionExample {

	// Path to your downloaded models
	private static final Path MODELS_DIR = Paths.get(System.getProperty("user.home"), "ai-models", "stable-diffusion-v3-5-medium");
	private static final Path OUTPUT_DIR = Paths.get("./generated_images");

	public static void main(String[] args) {
		System.out.println("Stable Diffusion v3.5 Medium Example");
		System.out.println("====================================");

		try {
			// Example 1: High Quality Generation (FP16 - best quality, slower)
			highQualityGeneration();

			// Example 2: Balanced Generation (Q8_0 - good quality, faster)
			balancedGeneration();

			// Example 3: Fast Generation (Q4_0 - lower quality, fastest)
			fastGeneration();

			// Example 4: Batch generation with different prompts
			batchGeneration();

		} catch (Exception e) {
			System.err.println("Error running Stable Diffusion examples: " + e.getMessage());
			e.printStackTrace();
		}
	}

	/**
	 * High quality generation using FP16 model - best quality but slower
	 */
	private static void highQualityGeneration() throws Exception {
		System.out.println("\n1. High Quality Generation (FP16)");
		System.out.println("---------------------------------");

		Path modelPath = MODELS_DIR.resolve("stable-diffusion-v3-5-medium-FP16.gguf");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(modelPath.toString())
				.outputDirectory(OUTPUT_DIR)
				.build()) {

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(1024)          // SD3.5 works well at 1024x1024
					.height(1024)
					.quality(ImageQuality.ULTRA)
					.steps(50)            // More steps = better quality
					.guidanceScale(7.0f)  // Good balance for SD3.5
					.styleHints(java.util.Arrays.asList("photorealistic", "highly detailed", "8k"))
					.negativePrompt("blurry, low quality, distorted, text, watermark")
					.build();

			String prompt = "A majestic snow-capped mountain peak at sunrise, with golden light illuminating the summit, pristine alpine lake in the foreground reflecting the mountain, surrounded by evergreen forests";

			System.out.println("Generating: " + prompt);
			GenerationResult result = converter.generateImage(prompt, params);

			if (result.isSuccess()) {
				System.out.println("✓ Generated: " + result.getOutputPath().get());
				System.out.println("  Size: " + result.getWidth() + "x" + result.getHeight());
				System.out.println("  Time: " + result.getGenerationTimeMs() + "ms");
				System.out.println("  Enhanced prompt: " + result.getOptimizedPrompt());
			} else {
				System.err.println("✗ Generation failed: " + result.getError().map(Exception::getMessage).orElse("Unknown error"));
			}
		}
	}

	/**
	 * Balanced generation using Q8_0 model - good quality and reasonable speed
	 */
	private static void balancedGeneration() throws Exception {
		System.out.println("\n2. Balanced Generation (Q8_0)");
		System.out.println("------------------------------");

		Path modelPath = MODELS_DIR.resolve("stable-diffusion-v3-5-medium-Q8_0.gguf");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(modelPath.toString())
				.outputDirectory(OUTPUT_DIR)
				.build()) {

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(768)
					.height(768)
					.quality(ImageQuality.HIGH)
					.steps(30)
					.guidanceScale(6.5f)
					.styleHints(java.util.Arrays.asList("digital art", "vibrant colors"))
					.build();

			String prompt = "A futuristic city skyline at night with neon lights, flying cars, and holographic advertisements, cyberpunk aesthetic";

			System.out.println("Generating: " + prompt);
			GenerationResult result = converter.generateImage(prompt, params);

			if (result.isSuccess()) {
				System.out.println("✓ Generated: " + result.getOutputPath().get());
				System.out.println("  Time: " + result.getGenerationTimeMs() + "ms");
			} else {
				System.err.println("✗ Generation failed: " + result.getError().map(Exception::getMessage).orElse("Unknown error"));
			}
		}
	}

	/**
	 * Fast generation using Q4_0 model - fastest but lower quality
	 */
	private static void fastGeneration() throws Exception {
		System.out.println("\n3. Fast Generation (Q4_0)");
		System.out.println("--------------------------");

		Path modelPath = MODELS_DIR.resolve("stable-diffusion-v3-5-medium-Q4_0.gguf");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(modelPath.toString())
				.outputDirectory(OUTPUT_DIR)
				.build()) {

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(512)
					.height(512)
					.quality(ImageQuality.STANDARD)
					.steps(20)            // Fewer steps for speed
					.guidanceScale(6.0f)
					.build();

			String prompt = "A cozy coffee shop interior with warm lighting, vintage furniture, and steaming coffee cups";

			System.out.println("Generating: " + prompt);
			GenerationResult result = converter.generateImage(prompt, params);

			if (result.isSuccess()) {
				System.out.println("✓ Generated: " + result.getOutputPath().get());
				System.out.println("  Time: " + result.getGenerationTimeMs() + "ms");
			} else {
				System.err.println("✗ Generation failed: " + result.getError().map(Exception::getMessage).orElse("Unknown error"));
			}
		}
	}

	/**
	 * Batch generation with multiple prompts
	 */
	private static void batchGeneration() throws Exception {
		System.out.println("\n4. Batch Generation");
		System.out.println("-------------------");

		Path modelPath = MODELS_DIR.resolve("stable-diffusion-v3-5-medium-Q8_0.gguf");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(modelPath.toString())
				.outputDirectory(OUTPUT_DIR)
				.batchSize(4)
				.build()) {

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(768)
					.height(768)
					.quality(ImageQuality.HIGH)
					.steps(25)
					.build();

			java.util.List<String> prompts = java.util.Arrays.asList(
					"A serene Japanese garden with cherry blossoms and a traditional wooden bridge",
					"An underwater coral reef teeming with colorful tropical fish",
					"A medieval castle perched on a cliff overlooking the ocean at sunset",
					"A steampunk airship floating through clouds above a Victorian city"
			);

			System.out.println("Processing " + prompts.size() + " prompts...");

			BatchGenerationResult batchResult = converter.generateBatch(prompts, params,
					progress -> System.out.printf("Progress: %.0f%%\n", progress * 100));

			if (batchResult.isSuccess()) {
				System.out.println("✓ Batch completed:");
				System.out.println("  Total: " + batchResult.getTotalPrompts());
				System.out.println("  Successful: " + batchResult.getSuccessfulGenerations());
				System.out.println("  Failed: " + batchResult.getFailedGenerations());
				System.out.println("  Success rate: " + String.format("%.1f%%", batchResult.getSuccessRate() * 100));
				System.out.println("  Total time: " + batchResult.getTotalTimeMs() + "ms");

				System.out.println("\nGenerated files:");
				batchResult.getResults().forEach(result -> {
					if (result.isSuccess()) {
						System.out.println("  ✓ " + result.getOutputPath().get());
					} else {
						System.out.println("  ✗ Failed: " + result.getOriginalPrompt());
					}
				});
			}
		}
	}

	/**
	 * Print usage information and model recommendations
	 */
	public static void printUsageInfo() {
		System.out.println("Model Selection Guide:");
		System.out.println("======================");
		System.out.println();
		System.out.println("FP16 Model:");
		System.out.println("  • Best quality, largest size (~10GB)");
		System.out.println("  • Use for final/production images");
		System.out.println("  • Requires more VRAM/RAM");
		System.out.println();
		System.out.println("Q8_0 Model:");
		System.out.println("  • Good quality, medium size (~5GB)");
		System.out.println("  • Best balance of quality/speed");
		System.out.println("  • Recommended for most use cases");
		System.out.println();
		System.out.println("Q4_0 Model:");
		System.out.println("  • Lower quality, smallest size (~2.5GB)");
		System.out.println("  • Fastest generation");
		System.out.println("  • Good for testing/prototyping");
		System.out.println();
		System.out.println("Recommended Settings for SD3.5:");
		System.out.println("  • Resolution: 1024x1024 (native), 768x768, or 512x512");
		System.out.println("  • Steps: 20-50 (more = better quality but slower)");
		System.out.println("  • Guidance Scale: 6.0-8.0 (higher = follows prompt more closely)");
		System.out.println("  • Use negative prompts to avoid unwanted elements");
	}
}