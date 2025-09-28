package de.kherud.llama.examples;

import de.kherud.llama.generation.TextToVisualConverter;
import de.kherud.llama.generation.TextToVisualConverterTypes.BatchGenerationResult;
import de.kherud.llama.generation.TextToVisualConverterTypes.GenerationResult;
import de.kherud.llama.generation.TextToVisualConverterTypes.ImageGenerationParameters;
import de.kherud.llama.generation.TextToVisualConverterTypes.LightingStyle;
import de.kherud.llama.generation.TextToVisualConverterTypes.MaterialQuality;
import de.kherud.llama.generation.TextToVisualConverterTypes.SceneComplexity;
import de.kherud.llama.generation.TextToVisualConverterTypes.SceneFormat;
import de.kherud.llama.generation.TextToVisualConverterTypes.SceneGenerationParameters;
import de.kherud.llama.generation.TextToVisualConverterTypes.SceneType;
import de.kherud.llama.generation.TextToVisualConverterTypes.VideoFormat;
import de.kherud.llama.generation.TextToVisualConverterTypes.VideoGenerationParameters;
import de.kherud.llama.generation.TextToVisualConverterTypes.VideoQuality;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Example demonstrating the updated TextToVisualConverter with Stable Diffusion integration.
 *
 * Prerequisites:
 * 1. Build stable-diffusion.cpp in /opt/stable-diffusion.cpp
 * 2. Have SD3.5 Medium models in ~/ai-models/stable-diffusion-v3-5-medium/
 */
public class TextToVisualConverterExample {

	private static final String MODEL_PATH = "dummy"; // Not used for SD integration
	private static final Path OUTPUT_DIR = Paths.get("./text_to_visual_output");

	public static void main(String[] args) {
		System.out.println("TextToVisualConverter with Stable Diffusion Integration");
		System.out.println("======================================================");
		System.out.println();

		// Check prerequisites
		if (!checkPrerequisites()) {
			return;
		}

		try {
			basicImageGeneration();
			advancedImageGeneration();
			videoGeneration();
			sceneGeneration();
			batchGeneration();
			asyncGeneration();
		} catch (Exception e) {
			System.err.println("Error running examples: " + e.getMessage());
			e.printStackTrace();
		}

		System.out.println();
		System.out.println("Example completed. Check output directory: " + OUTPUT_DIR);
	}

	private static boolean checkPrerequisites() {
		System.out.println("Checking prerequisites...");
		System.out.println("------------------------");

		// Check stable-diffusion.cpp executable
		Path sdPath = Paths.get("/opt/stable-diffusion.cpp/build/bin/sd");
		if (!Files.exists(sdPath)) {
			System.err.println("❌ stable-diffusion.cpp not found at: " + sdPath);
			System.err.println("   Please build it first:");
			System.err.println("   cd /opt/stable-diffusion.cpp");
			System.err.println("   mkdir build && cd build");
			System.err.println("   cmake .. -DGGML_CUDA=ON");
			System.err.println("   make -j8");
			return false;
		}
		System.out.println("✅ Found stable-diffusion.cpp executable");

		// Check models directory
		Path modelsDir = Paths.get(System.getProperty("user.home"), "ai-models", "stable-diffusion-v3-5-medium");
		if (!Files.exists(modelsDir)) {
			System.err.println("❌ Models directory not found: " + modelsDir);
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
			Path modelPath = modelsDir.resolve(model);
			if (Files.exists(modelPath)) {
				System.out.println("✅ Found model: " + model);
				foundModel = true;
			}
		}

		if (!foundModel) {
			System.err.println("❌ No SD3.5 Medium models found in " + modelsDir);
			return false;
		}

		System.out.println();
		return true;
	}

	private static void basicImageGeneration() throws Exception {
		System.out.println("1. Basic Image Generation");
		System.out.println("-------------------------");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(MODEL_PATH)
				.outputDirectory(OUTPUT_DIR.toString())
				.seed(42L)
				.build()) {

			String prompt = "A serene mountain landscape at sunrise with a crystal clear lake";

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(768)
					.height(768)
					.steps(25)
					.guidanceScale(7.0f)
					.styleHints(Arrays.asList("digital art", "high quality"))
					.build();

			System.out.println("Generating: " + prompt);
			GenerationResult result = converter.generateImage(prompt, params);

			if (result.isSuccess()) {
				System.out.println("✅ Image generated successfully!");
				System.out.println("   Output: " + result.getOutputPath());
				System.out.println("   Duration: " + result.getDuration());
				System.out.println("   Size: " + result.getWidth() + "x" + result.getHeight());
				System.out.println("   Original prompt: " + result.getOriginalPrompt());
				System.out.println("   Optimized prompt: " + result.getOptimizedPrompt());
			} else {
				System.err.println("❌ Generation failed: " + result.getError().map(Exception::getMessage).orElse("Unknown error"));
			}
		}
		System.out.println();
	}

	private static void advancedImageGeneration() throws Exception {
		System.out.println("2. Advanced Image Generation");
		System.out.println("----------------------------");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(MODEL_PATH)
				.outputDirectory(OUTPUT_DIR.toString())
				.build()) {

			String prompt = "A futuristic city with flying cars and neon lights, cyberpunk style, " +
					"detailed architecture, atmospheric lighting";

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(1024)
					.height(1024)
					.steps(40)
					.guidanceScale(8.0f)
					.styleHints(Arrays.asList("photorealistic", "detailed", "professional photography"))
					.build();

			System.out.println("Generating: " + prompt);
			GenerationResult result = converter.generateImage(prompt, params);

			if (result.isSuccess()) {
				System.out.println("✅ Advanced image generated successfully!");
				System.out.println("   Output: " + result.getOutputPath().orElse(null));
				System.out.println("   Duration: " + result.getDuration());
				System.out.println("   Steps: " + result.getGenerationSteps());
				System.out.println("   Guidance: " + result.getGuidanceScale());
				System.out.println("   Style hints: " + params.getStyleHints());
			} else {
				System.err.println("❌ Generation failed: " + result.getError().map(Exception::getMessage).orElse("Unknown error"));
			}
		}
		System.out.println();
	}

	private static void videoGeneration() throws Exception {
		System.out.println("\n3. Video Generation");
		System.out.println("-------------------");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(MODEL_PATH)
				.outputDirectory(OUTPUT_DIR)
				.build()) {

			String prompt = "A time-lapse of clouds moving across a blue sky";

			VideoGenerationParameters params = new VideoGenerationParameters.Builder()
					.width(640)
					.height(480)
					.duration(5.0f)
					.fps(24)
					.quality(VideoQuality.STANDARD)
					.format(VideoFormat.MP4)
					.build();

			GenerationResult result = converter.generateVideo(prompt, params);

			if (result.isSuccess()) {
				System.out.println("Generated video: " + result.getOutputPath());
				System.out.println("Duration: " + params.getDuration() + " seconds");
				System.out.println("FPS: " + params.getFps());
				System.out.println("Format: " + params.getFormat());
			}
		}
	}

	private static void sceneGeneration() throws Exception {
		System.out.println("\n4. 3D Scene Generation");
		System.out.println("-----------------------");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(MODEL_PATH)
				.outputDirectory(OUTPUT_DIR)
				.build()) {

			String prompt = "A medieval castle on a hill surrounded by trees";

			SceneGenerationParameters params = new SceneGenerationParameters.Builder()
					.sceneType(SceneType.LANDSCAPE)
					.complexity(SceneComplexity.MEDIUM)
					.lighting(LightingStyle.DRAMATIC)
					.materialQuality(MaterialQuality.HIGH)
					.outputFormat(SceneFormat.GLB)
					.build();

			GenerationResult result = converter.generate3DScene(prompt, params);

			if (result.isSuccess()) {
				System.out.println("Generated 3D scene: " + result.getOutputPath());
				System.out.println("Scene type: " + params.getSceneType());
				System.out.println("Complexity: " + params.getComplexity());
				System.out.println("Format: " + params.getOutputFormat());
			}
		}
	}

	private static void batchGeneration() throws Exception {
		System.out.println("5. Batch Generation");
		System.out.println("------------------");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(MODEL_PATH)
				.outputDirectory(OUTPUT_DIR.toString())
				.batchSize(3)
				.build()) {

			List<String> prompts = Arrays.asList(
					"A cute cat sleeping in a sunny garden",
					"A vintage car on a country road",
					"A steaming cup of coffee on a wooden table"
			);

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(512)
					.height(512)
					.steps(20)
					.guidanceScale(7.0f)
					.build();

			System.out.println("Generating " + prompts.size() + " images...");
			BatchGenerationResult result = converter.generateBatch(prompts, params, progress -> {
				System.out.printf("Progress: %.1f%%\n", progress * 100);
			});

			if (result.isSuccess()) {
				System.out.println("✅ Batch generation completed!");
				System.out.println("   Total duration: " + result.getDuration());
				System.out.println("   Results:");

				for (int i = 0; i < result.getResults().size(); i++) {
					GenerationResult singleResult = result.getResults().get(i);
					if (singleResult.isSuccess()) {
						System.out.println("     " + (i + 1) + ". ✅ " + singleResult.getOutputPath());
					} else {
						System.out.println("     " + (i + 1) + ". ❌ Failed: " +
								singleResult.getError().map(Exception::getMessage).orElse("Unknown error"));
					}
				}
			} else {
				System.err.println("❌ Batch generation failed");
			}
		}
		System.out.println();
	}

	private static void asyncGeneration() throws Exception {
		System.out.println("6. Asynchronous Generation");
		System.out.println("---------------------------");

		try (TextToVisualConverter converter = TextToVisualConverter.builder()
				.modelPath(MODEL_PATH)
				.outputDirectory(OUTPUT_DIR.toString())
				.build()) {

			String prompt1 = "A beautiful garden with colorful flowers";
			String prompt2 = "A sleek spaceship flying through space";

			ImageGenerationParameters params = new ImageGenerationParameters.Builder()
					.width(512)
					.height(512)
					.steps(25)
					.guidanceScale(7.0f)
					.build();

			System.out.println("Starting async generations...");

			CompletableFuture<GenerationResult> future1 = converter.generateImageAsync(prompt1, params);
			CompletableFuture<GenerationResult> future2 = converter.generateImageAsync(prompt2, params);

			CompletableFuture.allOf(future1, future2).thenRun(() -> {
				System.out.println("Both async generations completed!");
			});

			GenerationResult result1 = future1.get();
			GenerationResult result2 = future2.get();

			System.out.println("Async generation 1: " +
					(result1.isSuccess() ? result1.getOutputPath() : "Failed"));
			System.out.println("Async generation 2: " +
					(result2.isSuccess() ? result2.getOutputPath() : "Failed"));
		}
		System.out.println();
	}
}
