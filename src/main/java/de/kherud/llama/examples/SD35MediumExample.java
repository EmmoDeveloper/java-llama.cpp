package de.kherud.llama.examples;

import de.kherud.llama.diffusion.StableDiffusionCppWrapper;
import de.kherud.llama.diffusion.StableDiffusionCppWrapper.GenerationParams;
import de.kherud.llama.diffusion.StableDiffusionCppWrapper.GenerationResult;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Example demonstrating how to use Stable Diffusion 3.5 Medium with stable-diffusion.cpp.
 *
 * Prerequisites:
 * 1. Build stable-diffusion.cpp in /opt/stable-diffusion.cpp
 * 2. Have SD3.5 Medium models in ~/ai-models/stable-diffusion-v3-5-medium/
 * 3. Download text encoders (CLIP-L, CLIP-G, T5XXL) if needed
 *
 * Build stable-diffusion.cpp:
 *   cd /opt/stable-diffusion.cpp
 *   mkdir build
 *   cd build
 *   cmake .. -DGGML_CUDA=ON  # or -DGGML_METAL=ON for Mac
 *   make -j8
 *
 * The executable will be at: /opt/stable-diffusion.cpp/build/bin/sd
 */
public class SD35MediumExample {

	private static final Path MODELS_DIR = Paths.get(System.getProperty("user.home"),
		"ai-models", "stable-diffusion-v3-5-medium");
	private static final Path OUTPUT_DIR = Paths.get("./sd35_images");
	private static final String SD_EXECUTABLE = "/opt/stable-diffusion.cpp/build/bin/sd";

	public static void main(String[] args) {
		System.out.println("Stable Diffusion 3.5 Medium Example");
		System.out.println("===================================");
		System.out.println();

		// Check if stable-diffusion.cpp is available
		checkPrerequisites();

		// Run examples with different models
		runWithQ4Model();
		runWithQ8Model();
		runWithFP16Model();

		// Show how to generate with specific parameters
		runCustomGeneration();
	}

	private static void checkPrerequisites() {
		System.out.println("Checking prerequisites...");
		System.out.println("------------------------");

		// Check if stable-diffusion.cpp executable exists
		Path sdPath = Paths.get(SD_EXECUTABLE);
		if (!sdPath.toFile().exists()) {
			System.err.println("❌ stable-diffusion.cpp not found at: " + SD_EXECUTABLE);
			System.err.println("   Please build it first:");
			System.err.println("   cd /opt/stable-diffusion.cpp");
			System.err.println("   mkdir build && cd build");
			System.err.println("   cmake .. -DGGML_CUDA=ON");
			System.err.println("   make -j8");
			return;
		}
		System.out.println("✅ Found stable-diffusion.cpp executable");

		// Check models
		String[] models = {
			"stable-diffusion-v3-5-medium-Q4_0.gguf",
			"stable-diffusion-v3-5-medium-Q8_0.gguf",
			"stable-diffusion-v3-5-medium-FP16.gguf"
		};

		for (String model : models) {
			Path modelPath = MODELS_DIR.resolve(model);
			if (modelPath.toFile().exists()) {
				System.out.println("✅ Found model: " + model);
			} else {
				System.out.println("⚠️  Missing model: " + model);
			}
		}

		System.out.println();
		System.out.println("Note: SD3.5 Medium may also need text encoders:");
		System.out.println("  - CLIP-L: for text encoding");
		System.out.println("  - CLIP-G: for text encoding");
		System.out.println("  - T5XXL: for enhanced text understanding");
		System.out.println("  Download from: https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/tree/main/text_encoders");
		System.out.println();
	}

	private static void runWithQ4Model() {
		System.out.println("1. Generating with Q4_0 Model (Fast)");
		System.out.println("------------------------------------");

		StableDiffusionCppWrapper sd = StableDiffusionCppWrapper.builder()
				.executable(SD_EXECUTABLE)
				.model(MODELS_DIR.resolve("stable-diffusion-v3-5-medium-Q4_0.gguf").toString())
				// Add text encoders if you have them:
				// .clipL("/path/to/clip_l.safetensors")
				// .clipG("/path/to/clip_g.safetensors")
				// .t5xxl("/path/to/t5xxl_fp16.safetensors")
				.outputDirectory(OUTPUT_DIR)
				.build();

		GenerationParams params = new GenerationParams();
		params.prompt = "A serene mountain landscape at sunrise with a crystal clear lake";
		params.negativePrompt = "blurry, low quality, distorted";
		params.width = 512;
		params.height = 512;
		params.steps = 20;
		params.cfgScale = 7.0f;
		params.slgScale = 2.5f;  // Good for SD3.5 Medium
		params.samplingMethod = "euler";
		params.verbose = true;

		System.out.println("Generating: " + params.prompt);
		GenerationResult result = sd.generateImage(params);

		if (result.success) {
			System.out.println("✅ Image generated: " + result.imagePath);
		} else {
			System.err.println("❌ Generation failed: " + result.message);
		}
		System.out.println();
	}

	private static void runWithQ8Model() {
		System.out.println("2. Generating with Q8_0 Model (Balanced)");
		System.out.println("----------------------------------------");

		StableDiffusionCppWrapper sd = StableDiffusionCppWrapper.builder()
				.executable(SD_EXECUTABLE)
				.model(MODELS_DIR.resolve("stable-diffusion-v3-5-medium-Q8_0.gguf").toString())
				.outputDirectory(OUTPUT_DIR)
				.build();

		GenerationParams params = new GenerationParams();
		params.prompt = "A futuristic city with flying cars and neon lights, cyberpunk style";
		params.width = 768;
		params.height = 768;
		params.steps = 30;
		params.cfgScale = 7.5f;
		params.slgScale = 2.5f;
		params.verbose = true;

		System.out.println("Generating: " + params.prompt);
		GenerationResult result = sd.generateImage(params);

		if (result.success) {
			System.out.println("✅ Image generated: " + result.imagePath);
		} else {
			System.err.println("❌ Generation failed: " + result.message);
		}
		System.out.println();
	}

	private static void runWithFP16Model() {
		System.out.println("3. Generating with FP16 Model (High Quality)");
		System.out.println("--------------------------------------------");

		StableDiffusionCppWrapper sd = StableDiffusionCppWrapper.builder()
				.executable(SD_EXECUTABLE)
				.model(MODELS_DIR.resolve("stable-diffusion-v3-5-medium-FP16.gguf").toString())
				.outputDirectory(OUTPUT_DIR)
				.build();

		GenerationParams params = new GenerationParams();
		params.prompt = "A majestic dragon soaring through clouds at sunset, highly detailed, fantasy art";
		params.width = 1024;
		params.height = 1024;
		params.steps = 50;
		params.cfgScale = 8.0f;
		params.slgScale = 2.5f;
		params.verbose = true;

		System.out.println("Generating: " + params.prompt);
		GenerationResult result = sd.generateImage(params);

		if (result.success) {
			System.out.println("✅ Image generated: " + result.imagePath);
		} else {
			System.err.println("❌ Generation failed: " + result.message);
		}
		System.out.println();
	}

	private static void runCustomGeneration() {
		System.out.println("4. Custom Generation Example");
		System.out.println("----------------------------");

		// Using Q8 model for balance
		StableDiffusionCppWrapper sd = StableDiffusionCppWrapper.builder()
				.executable(SD_EXECUTABLE)
				.model(MODELS_DIR.resolve("stable-diffusion-v3-5-medium-Q8_0.gguf").toString())
				.outputDirectory(OUTPUT_DIR)
				.build();

		// Create custom parameters
		GenerationParams params = new GenerationParams();
		params.prompt = "A cozy coffee shop interior with warm lighting, vintage furniture, " +
						"steaming coffee cups, books on shelves, photorealistic";
		params.negativePrompt = "blurry, low quality, cartoon, anime, distorted, ugly";
		params.width = 768;
		params.height = 768;
		params.steps = 35;
		params.cfgScale = 7.0f;
		params.slgScale = 2.5f;  // Skip Layer Guidance - good for SD3.5
		params.samplingMethod = "euler";  // or try "dpm++2m" for different results
		params.seed = 42;  // Fixed seed for reproducibility
		params.clipOnCpu = true;  // Save GPU memory by running CLIP on CPU
		params.verbose = false;  // Less output

		System.out.println("Custom generation:");
		System.out.println("  Prompt: " + params.prompt);
		System.out.println("  Size: " + params.width + "x" + params.height);
		System.out.println("  Steps: " + params.steps);
		System.out.println("  CFG Scale: " + params.cfgScale);
		System.out.println("  SLG Scale: " + params.slgScale);
		System.out.println("  Seed: " + params.seed);

		GenerationResult result = sd.generateImage(params);

		if (result.success) {
			System.out.println("✅ Image generated: " + result.imagePath);
		} else {
			System.err.println("❌ Generation failed: " + result.message);
		}
	}

	/**
	 * Print recommended settings for SD3.5 Medium
	 */
	public static void printRecommendedSettings() {
		System.out.println();
		System.out.println("Recommended Settings for SD3.5 Medium:");
		System.out.println("======================================");
		System.out.println();
		System.out.println("Resolution:");
		System.out.println("  • 512x512: Fast, lower quality");
		System.out.println("  • 768x768: Good balance (recommended)");
		System.out.println("  • 1024x1024: High quality, slower");
		System.out.println();
		System.out.println("Steps:");
		System.out.println("  • 20-30: Fast generation");
		System.out.println("  • 30-50: Good quality (recommended)");
		System.out.println("  • 50+: Maximum quality");
		System.out.println();
		System.out.println("CFG Scale:");
		System.out.println("  • 5-7: More creative");
		System.out.println("  • 7-8: Balanced (recommended)");
		System.out.println("  • 8-10: Strict prompt following");
		System.out.println();
		System.out.println("SLG Scale (Skip Layer Guidance):");
		System.out.println("  • 0: Disabled");
		System.out.println("  • 2.5: Good for SD3.5 Medium (recommended)");
		System.out.println("  • Higher: More guidance, may reduce creativity");
		System.out.println();
		System.out.println("Sampling Methods:");
		System.out.println("  • euler: Default for SD3.5");
		System.out.println("  • euler_a: Alternative Euler");
		System.out.println("  • dpm++2m: Often good results");
		System.out.println("  • dpm++2mv2: Updated version");
	}
}