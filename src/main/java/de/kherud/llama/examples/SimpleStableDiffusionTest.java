package de.kherud.llama.examples;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaIterable;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Simple test to verify Stable Diffusion model loading and basic functionality.
 *
 * This demonstrates:
 * 1. Loading a Stable Diffusion model from your ~/ai-models directory
 * 2. Verifying it's detected as a diffusion model
 * 3. Testing basic model functionality
 */
public class SimpleStableDiffusionTest {

	public static void main(String[] args) {
		System.out.println("Simple Stable Diffusion Test");
		System.out.println("============================");

		// Test all available models
		testModel("stable-diffusion-v3-5-medium-Q4_0.gguf", "Fast Q4_0 model");
		testModel("stable-diffusion-v3-5-medium-Q8_0.gguf", "Balanced Q8_0 model");
		testModel("stable-diffusion-v3-5-medium-FP16.gguf", "High quality FP16 model");
	}

	private static void testModel(String modelFileName, String description) {
		System.out.println("\nTesting: " + description);
		System.out.println("File: " + modelFileName);
		System.out.println("----------------------------------------");

		Path modelPath = Paths.get(System.getProperty("user.home"),
			"ai-models", "stable-diffusion-v3-5-medium", modelFileName);

		// Check if model file exists
		if (!modelPath.toFile().exists()) {
			System.out.println("⚠️  Model not found: " + modelPath);
			return;
		}

		try {
			// Configure model parameters for diffusion models
			ModelParameters modelParams = new ModelParameters()
					.setModel(modelPath.toString())
					.setCtxSize(4096)        // Context size for diffusion
					.setGpuLayers(40)        // Use GPU if available
					.setThreads(8);          // CPU threads

			System.out.println("Loading model...");
			long startTime = System.currentTimeMillis();

			try (LlamaModel model = new LlamaModel(modelParams)) {
				long loadTime = System.currentTimeMillis() - startTime;
				System.out.println("✅ Model loaded successfully in " + loadTime + "ms");

				// Check if it's a diffusion model
				boolean isDiffusion = model.isDiffusionModel();
				System.out.println("Is diffusion model: " + (isDiffusion ? "✅ YES" : "❌ NO"));

				if (isDiffusion) {
					System.out.println("✅ This is a valid Stable Diffusion model!");

					// Get model info
					System.out.println("Model info:");
					System.out.println("  - Context size: " + model.getContextSize());
					System.out.println("  - Vocabulary size: " + model.getVocabularySize());
					System.out.println("  - Model size: " + model.getModelSize() + " bytes");

					// Test basic inference (this might not generate actual images yet)
					testBasicInference(model);
				} else {
					System.out.println("❌ This model is not recognized as a diffusion model");
				}

			} catch (Exception e) {
				System.err.println("❌ Error using model: " + e.getMessage());
			}

		} catch (Exception e) {
			System.err.println("❌ Error loading model: " + e.getMessage());
		}
	}

	private static void testBasicInference(LlamaModel model) {
		try {
			System.out.println("\nTesting basic inference...");

			// Simple test prompt
			String prompt = "A beautiful landscape";

			InferenceParameters params = new InferenceParameters(prompt)
					.setNPredict(50)         // Limited output for testing
					.setTemperature(0.7f)    // Some randomness
					.setTopP(0.9f)           // Nucleus sampling
					.setTopK(40);            // Top-k sampling

			System.out.println("Prompt: " + prompt);
			System.out.println("Generating...");

			LlamaIterable outputs = model.generate(params);
			StringBuilder response = new StringBuilder();
			int tokenCount = 0;

			for (LlamaOutput output : outputs) {
				response.append(output.text);
				tokenCount++;
				if (tokenCount > 10) break; // Limit for testing
			}

			System.out.println("Generated " + tokenCount + " tokens");
			System.out.println("Response: " + response.toString().trim());
			System.out.println("✅ Basic inference test completed");

		} catch (Exception e) {
			System.err.println("❌ Error during inference: " + e.getMessage());
		}
	}

	/**
	 * Print information about using these models
	 */
	public static void printUsageInfo() {
		System.out.println("\n" + "=".repeat(60));
		System.out.println("How to Use Your Stable Diffusion Models");
		System.out.println("=".repeat(60));
		System.out.println();
		System.out.println("You have these Stable Diffusion v3.5 Medium models:");
		System.out.println();
		System.out.println("1. FP16 (~10GB) - Highest quality");
		System.out.println("   • Best for final/production images");
		System.out.println("   • Requires more memory");
		System.out.println();
		System.out.println("2. Q8_0 (~5GB) - Good quality, balanced");
		System.out.println("   • Recommended for most use cases");
		System.out.println("   • Good quality/speed tradeoff");
		System.out.println();
		System.out.println("3. Q4_0 (~2.5GB) - Fast, lower quality");
		System.out.println("   • Good for testing and quick iterations");
		System.out.println("   • Lowest memory requirements");
		System.out.println();
		System.out.println("To generate images:");
		System.out.println("1. Use TextToVisualConverter with your model path");
		System.out.println("2. Set appropriate width/height (512, 768, or 1024)");
		System.out.println("3. Use guidance scale 6.0-8.0 for good prompt following");
		System.out.println("4. Use 20-50 steps (more = better quality)");
		System.out.println();
		System.out.println("Example code:");
		System.out.println("  TextToVisualConverter converter = TextToVisualConverter.builder()");
		System.out.println("      .modelPath(\"~/ai-models/stable-diffusion-v3-5-medium/stable-diffusion-v3-5-medium-Q8_0.gguf\")");
		System.out.println("      .outputDirectory(\"./images\")");
		System.out.println("      .build();");
		System.out.println();
		System.out.println("  ImageGenerationParameters params = new ImageGenerationParameters.Builder()");
		System.out.println("      .width(768).height(768)");
		System.out.println("      .steps(30)");
		System.out.println("      .guidanceScale(7.0f)");
		System.out.println("      .build();");
		System.out.println();
		System.out.println("  GenerationResult result = converter.generateImage(\"your prompt here\", params);");
	}
}
