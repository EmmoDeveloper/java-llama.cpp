package de.kherud.llama.examples;

import de.kherud.llama.diffusion.NativeStableDiffusion;
import de.kherud.llama.diffusion.NativeStableDiffusionWrapper;
import de.kherud.llama.diffusion.StableDiffusionResult;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Example demonstrating native Stable Diffusion integration.
 *
 * Prerequisites:
 * 1. Build the jllama library with stable-diffusion.cpp support
 * 2. Have SD3.5 Medium models in ~/ai-models/stable-diffusion-v3-5-medium/
 */
public class NativeStableDiffusionExample {

	private static final Path MODELS_DIR = Paths.get(System.getProperty("user.home"),
		"ai-models", "stable-diffusion-v3-5-medium");
	private static final Path OUTPUT_DIR = Paths.get("./native_sd_images");

	public static void main(String[] args) {
		System.out.println("Native Stable Diffusion Integration Example");
		System.out.println("==========================================");
		System.out.println();

		// Check prerequisites
		if (!checkPrerequisites()) {
			return;
		}

		// Run examples
		runBasicGeneration();
		runAdvancedGeneration();
		runAutoDetectionExample();

		System.out.println();
		System.out.println("Example completed. Check output directory: " + OUTPUT_DIR);
	}

	private static boolean checkPrerequisites() {
		System.out.println("Checking prerequisites...");
		System.out.println("------------------------");

		// Check models directory
		if (!Files.exists(MODELS_DIR)) {
			System.err.println("❌ Models directory not found: " + MODELS_DIR);
			return false;
		}

		// Check for at least one model
		String[] models = {
			"stable-diffusion-v3-5-medium-FP16.gguf",
			"stable-diffusion-v3-5-medium-Q8_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_1.gguf"
		};

		boolean foundModel = false;
		for (String model : models) {
			Path modelPath = MODELS_DIR.resolve(model);
			if (Files.exists(modelPath)) {
				System.out.println("✅ Found model: " + model);
				foundModel = true;
			}
		}

		if (!foundModel) {
			System.err.println("❌ No SD3.5 Medium models found in " + MODELS_DIR);
			return false;
		}

		// Test native library loading
		try {
			String sysInfo = NativeStableDiffusion.getSystemInfo();
			System.out.println("✅ Native library loaded successfully");
			System.out.println("   System info: " + sysInfo.split("\n")[0]); // First line only
		} catch (Exception e) {
			System.err.println("❌ Failed to load native library: " + e.getMessage());
			return false;
		}

		// Create output directory
		try {
			Files.createDirectories(OUTPUT_DIR);
			System.out.println("✅ Output directory ready: " + OUTPUT_DIR);
		} catch (Exception e) {
			System.err.println("❌ Failed to create output directory: " + e.getMessage());
			return false;
		}

		System.out.println();
		return true;
	}

	private static void runBasicGeneration() {
		System.out.println("1. Basic Native Generation");
		System.out.println("--------------------------");

		// Find first available model
		String modelPath = findFirstAvailableModel();
		if (modelPath == null) {
			System.err.println("❌ No models available");
			return;
		}

		try (NativeStableDiffusionWrapper sd = NativeStableDiffusionWrapper.builder()
				.modelPath(modelPath)
				.build()) {

			String prompt = "A serene mountain landscape at sunrise with a crystal clear lake";
			System.out.println("Generating: " + prompt);

			StableDiffusionResult result = sd.generateImage(prompt);

			if (result.isSuccess()) {
				System.out.println("✅ Image generated successfully!");
				System.out.println("   Size: " + result.getWidth() + "x" + result.getHeight());
				System.out.println("   Channels: " + result.getChannels());
				System.out.println("   Time: " + result.getGenerationTime() + "s");
				System.out.println("   Data size: " + result.getImageDataSize() + " bytes");

				// Save the image
				Path outputPath = OUTPUT_DIR.resolve("basic_generation.png");
				try {
					NativeStableDiffusionWrapper.saveImageAsPng(result, outputPath);
					System.out.println("   Saved to: " + outputPath);
				} catch (Exception e) {
					System.err.println("   Failed to save image: " + e.getMessage());
				}
			} else {
				System.err.println("❌ Generation failed: " + result.getErrorMessage().orElse("Unknown error"));
			}

		} catch (Exception e) {
			System.err.println("❌ Error: " + e.getMessage());
			e.printStackTrace();
		}
		System.out.println();
	}

	private static void runAdvancedGeneration() {
		System.out.println("2. Advanced Native Generation");
		System.out.println("-----------------------------");

		String modelPath = findFirstAvailableModel();
		if (modelPath == null) {
			System.err.println("❌ No models available");
			return;
		}

		try (NativeStableDiffusionWrapper sd = NativeStableDiffusionWrapper.builder()
				.modelPath(modelPath)
				.keepClipOnCpu(true)
				.build()) {

			NativeStableDiffusionWrapper.GenerationParameters params =
				NativeStableDiffusionWrapper.GenerationParameters.forSD35Medium()
					.withPrompt("A futuristic city with flying cars and neon lights, cyberpunk style, detailed architecture")
					.withNegativePrompt("blurry, low quality, distorted, ugly, cartoon")
					.withSize(1024, 1024)
					.withSteps(40)
					.withCfgScale(8.0f)
					.withSeed(42)
					.withSampleMethod(NativeStableDiffusion.SAMPLE_METHOD_DPMPP2M);

			System.out.println("Generating advanced image...");
			System.out.println("  Size: " + params.width + "x" + params.height);
			System.out.println("  Steps: " + params.steps);
			System.out.println("  CFG Scale: " + params.cfgScale);
			System.out.println("  SLG Scale: " + params.slgScale);
			System.out.println("  Sample Method: " + NativeStableDiffusion.getSampleMethodName(params.sampleMethod));
			System.out.println("  Seed: " + params.seed);

			StableDiffusionResult result = sd.generateImage(params);

			if (result.isSuccess()) {
				System.out.println("✅ Advanced image generated successfully!");
				System.out.println("   Size: " + result.getWidth() + "x" + result.getHeight());
				System.out.println("   Time: " + result.getGenerationTime() + "s");

				Path outputPath = OUTPUT_DIR.resolve("advanced_generation.png");
				try {
					NativeStableDiffusionWrapper.saveImageAsPng(result, outputPath);
					System.out.println("   Saved to: " + outputPath);
				} catch (Exception e) {
					System.err.println("   Failed to save image: " + e.getMessage());
				}
			} else {
				System.err.println("❌ Advanced generation failed: " + result.getErrorMessage().orElse("Unknown error"));
			}

		} catch (Exception e) {
			System.err.println("❌ Error: " + e.getMessage());
			e.printStackTrace();
		}
		System.out.println();
	}

	private static void runAutoDetectionExample() {
		System.out.println("3. Auto-Detection Example");
		System.out.println("-------------------------");

		try {
			var sdOpt = NativeStableDiffusionWrapper.createWithAutoDetection();
			if (sdOpt.isPresent()) {
				try (NativeStableDiffusionWrapper sd = sdOpt.get()) {
					System.out.println("✅ Auto-detected model: " + sd.getModelPath());

					String prompt = "A cute cat sleeping in a sunny garden";
					StableDiffusionResult result = sd.generateImage(prompt, 512, 512);

					if (result.isSuccess()) {
						System.out.println("✅ Auto-detection generation successful!");
						System.out.println("   Time: " + result.getGenerationTime() + "s");

						Path outputPath = OUTPUT_DIR.resolve("auto_detection.png");
						try {
							NativeStableDiffusionWrapper.saveImageAsPng(result, outputPath);
							System.out.println("   Saved to: " + outputPath);
						} catch (Exception e) {
							System.err.println("   Failed to save image: " + e.getMessage());
						}
					} else {
						System.err.println("❌ Auto-detection generation failed: " + result.getErrorMessage().orElse("Unknown error"));
					}
				}
			} else {
				System.err.println("❌ No models found for auto-detection");
			}
		} catch (Exception e) {
			System.err.println("❌ Auto-detection error: " + e.getMessage());
			e.printStackTrace();
		}
		System.out.println();
	}

	private static String findFirstAvailableModel() {
		String[] models = {
			"stable-diffusion-v3-5-medium-FP16.gguf",
			"stable-diffusion-v3-5-medium-Q8_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_1.gguf"
		};

		for (String model : models) {
			Path modelPath = MODELS_DIR.resolve(model);
			if (Files.exists(modelPath)) {
				return modelPath.toString();
			}
		}
		return null;
	}

	/**
	 * Print performance tips and recommendations
	 */
	public static void printPerformanceTips() {
		System.out.println();
		System.out.println("Performance Tips for Native Integration:");
		System.out.println("======================================");
		System.out.println();
		System.out.println("Model Selection:");
		System.out.println("  • FP16: Highest quality, requires more VRAM");
		System.out.println("  • Q8_0: Good balance of quality and speed (recommended)");
		System.out.println("  • Q4_0: Fastest generation, lower quality");
		System.out.println();
		System.out.println("Memory Management:");
		System.out.println("  • Use keepClipOnCpu(true) to save GPU memory");
		System.out.println("  • Close wrappers when done to free native resources");
		System.out.println("  • Consider smaller image sizes for faster generation");
		System.out.println();
		System.out.println("SD3.5 Medium Optimization:");
		System.out.println("  • SLG Scale: 2.5 is optimal for SD3.5 Medium");
		System.out.println("  • CFG Scale: 7-8 for balanced prompt following");
		System.out.println("  • Steps: 30-40 for good quality");
		System.out.println("  • Sample Method: Euler or DPM++2M work well");
	}
}