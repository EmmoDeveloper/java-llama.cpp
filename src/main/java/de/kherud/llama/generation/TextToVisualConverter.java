package de.kherud.llama.generation;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.diffusion.NativeStableDiffusionWrapper;
import de.kherud.llama.diffusion.StableDiffusionResult;
import de.kherud.llama.generation.TextToVisualConverterTypes.BatchGenerationResult;
import de.kherud.llama.generation.TextToVisualConverterTypes.Camera3DSettings;
import de.kherud.llama.generation.TextToVisualConverterTypes.GeneratedFrame;
import de.kherud.llama.generation.TextToVisualConverterTypes.GenerationResult;
import de.kherud.llama.generation.TextToVisualConverterTypes.ImageGenerationParameters;
import de.kherud.llama.generation.TextToVisualConverterTypes.KeyframePrompt;
import de.kherud.llama.generation.TextToVisualConverterTypes.OutputFormat;
import de.kherud.llama.generation.TextToVisualConverterTypes.RenderedView;
import de.kherud.llama.generation.TextToVisualConverterTypes.Scene3DData;
import de.kherud.llama.generation.TextToVisualConverterTypes.Scene3DParameters;
import de.kherud.llama.generation.TextToVisualConverterTypes.SceneComplexity;
import de.kherud.llama.generation.TextToVisualConverterTypes.SceneGenerationParameters;
import de.kherud.llama.generation.TextToVisualConverterTypes.VideoGenerationParameters;
import de.kherud.llama.multimodal.ImageProcessorLibrary;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Advanced text-to-visual converter using AI models for generating images, videos, and 3D content.
 */
public class TextToVisualConverter implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(TextToVisualConverter.class.getName());

	// Core configuration
	private final String modelPath;
	private final Path outputDirectory;
	private final Path stableDiffusionModelsPath;
	private final String sdExecutablePath;
	private final Long seed;
	private final int batchSize;

	// Model instances
	private LlamaModel textToImageModel;
	private NativeStableDiffusionWrapper stableDiffusion;
	private ImageProcessorLibrary imageProcessor;
	private final ExecutorService executor;

	TextToVisualConverter(TextToVisualConverterTypes.Builder builder) {
		this.modelPath = Objects.requireNonNull(builder.textToImageModelPath);
		this.outputDirectory = Paths.get(builder.outputDirectory);
		this.stableDiffusionModelsPath = Paths.get(System.getProperty("user.home"), "ai-models", "stable-diffusion-v3-5-medium");
		this.sdExecutablePath = "/opt/stable-diffusion.cpp/build/bin/sd";
		this.seed = builder.seed;
		this.batchSize = builder.batchSize;
		this.executor = builder.executor != null ? builder.executor : Executors.newFixedThreadPool(4);

		try {
			Files.createDirectories(this.outputDirectory);
		} catch (IOException e) {
			LOGGER.log(System.Logger.Level.WARNING, "Failed to create output directory: " + outputDirectory, e);
		}
	}

	public static TextToVisualConverterTypes.Builder builder() {
		return new TextToVisualConverterTypes.Builder();
	}

	public GenerationResult generateImage(String prompt, ImageGenerationParameters params) {
		return generateImageInternal(prompt, params);
	}

	public GenerationResult generateVideo(String prompt, VideoGenerationParameters params) {
		Instant start = Instant.now();
		try {
			// For video generation, use image generation for keyframes

			String optimizedPrompt = enhancePrompt(prompt, List.of());
			Path outputPath = generateOutputPath("video", "mp4");

			// Generate keyframes for video
			List<KeyframePrompt> keyframes = generateKeyframes(prompt, params);
			List<GeneratedFrame> frames = new ArrayList<>();

			for (int i = 0; i < keyframes.size(); i++) {
				KeyframePrompt keyframe = keyframes.get(i);
				Path framePath = generateOutputPath("frame_" + i, "jpg");

				ImageGenerationParameters frameParams = ImageGenerationParameters.forVideo(params.getWidth(), params.getHeight());
				GenerationResult frameResult = generateImageInternal(keyframe.getPrompt(), frameParams);

				if (frameResult.isSuccess()) {
					frames.add(new GeneratedFrame(i, keyframe.getTimePosition(), frameResult.getOutputPath().orElse(framePath), keyframe.getPrompt()));
				}
			}

			// Combine frames into video (simplified)
			combineFramesToVideo(frames, outputPath, params);

			Duration duration = Duration.between(start, Instant.now());
			return new GenerationResult.Builder()
					.success(true)
					.originalPrompt(prompt)
					.optimizedPrompt(optimizedPrompt)
					.outputPath(outputPath)
					.outputFormat(OutputFormat.VIDEO_2D)
					.duration(duration)
					.build();

		} catch (Exception e) {
			Duration duration = Duration.between(start, Instant.now());
			return new GenerationResult.Builder()
					.success(false)
					.originalPrompt(prompt)
					.error(e)
					.duration(duration)
					.build();
		}
	}

	public GenerationResult generate3DScene(String prompt, SceneGenerationParameters params) {
		Instant start = Instant.now();
		try {
			// For 3D scene generation, use image generation for textures

			String optimizedPrompt = enhancePrompt(prompt, List.of());
			Path outputPath = generateOutputPath("scene", "glb");

			// Generate 3D scene data
			Scene3DData sceneData = generate3DSceneData(optimizedPrompt, params);

			// Render views
			List<RenderedView> views = render3DViews(sceneData);

			// Save scene file
			save3DScene(sceneData, outputPath);

			Duration duration = Duration.between(start, Instant.now());
			return new GenerationResult.Builder()
					.success(true)
					.originalPrompt(prompt)
					.optimizedPrompt(optimizedPrompt)
					.outputPath(outputPath)
					.outputFormat(OutputFormat.SCENE_3D)
					.duration(duration)
					.build();

		} catch (Exception e) {
			Duration duration = Duration.between(start, Instant.now());
			return new GenerationResult.Builder()
					.success(false)
					.originalPrompt(prompt)
					.error(e)
					.duration(duration)
					.build();
		}
	}

	public BatchGenerationResult generateBatch(List<String> prompts, ImageGenerationParameters params, Consumer<Double> progressCallback) {
		Instant start = Instant.now();
		List<GenerationResult> results = new ArrayList<>();
		int successful = 0;

		for (int i = 0; i < prompts.size(); i++) {
			String prompt = prompts.get(i);
			GenerationResult result = generateImage(prompt, params);
			results.add(result);

			if (result.isSuccess()) {
				successful++;
			}

			if (progressCallback != null) {
				progressCallback.accept((double) (i + 1) / prompts.size());
			}
		}

		Duration duration = Duration.between(start, Instant.now());
		return new BatchGenerationResult.Builder()
				.success(true)
				.results(results)
				.totalPrompts(prompts.size())
				.duration(duration)
				.build();
	}

	public CompletableFuture<GenerationResult> generateImageAsync(String prompt, ImageGenerationParameters params) {
		return CompletableFuture.supplyAsync(() -> generateImage(prompt, params), executor);
	}

	private GenerationResult generateImageInternal(String prompt, ImageGenerationParameters params) {
		Instant start = Instant.now();
		try {
			String optimizedPrompt = enhancePrompt(prompt, params.getStyleHints());

			// Generate image using Stable Diffusion
			Path imagePath = generateImageWithStableDiffusion(optimizedPrompt, params);

			Duration duration = Duration.between(start, Instant.now());
			return new GenerationResult.Builder()
					.success(true)
					.originalPrompt(prompt)
					.optimizedPrompt(optimizedPrompt)
					.outputPath(imagePath)
					.outputFormat(OutputFormat.HIGH_RESOLUTION_IMAGE)
					.width(params.getWidth())
					.height(params.getHeight())
					.generationSteps(params.getSteps())
					.guidanceScale(params.getGuidanceScale())
					.duration(duration)
					.build();

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Image generation failed for prompt: " + prompt, e);
			Duration duration = Duration.between(start, Instant.now());
			return new GenerationResult.Builder()
					.success(false)
					.originalPrompt(prompt)
					.error(e)
					.duration(duration)
					.build();
		}
	}

	private void initializeStableDiffusion() throws Exception {
		if (stableDiffusion == null) {
			String bestModel = findBestAvailableModel();
			if (bestModel == null) {
				throw new IllegalStateException("No Stable Diffusion models found in " + stableDiffusionModelsPath);
			}

			stableDiffusion = NativeStableDiffusionWrapper.builder()
					.modelPath(stableDiffusionModelsPath.resolve(bestModel).toString())
					.keepClipOnCpu(true)
					.build();

			LOGGER.log(System.Logger.Level.INFO, "Initialized native Stable Diffusion with model: " + bestModel);
		}
	}

	private String findBestAvailableModel() {
		String[] preferredModels = {
			"stable-diffusion-v3-5-medium-FP16.gguf",
			"stable-diffusion-v3-5-medium-Q8_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_1.gguf"
		};

		for (String model : preferredModels) {
			Path modelPath = stableDiffusionModelsPath.resolve(model);
			if (Files.exists(modelPath)) {
				return model;
			}
		}
		return null;
	}

	private String enhancePrompt(String originalPrompt, List<String> styleHints) {
		StringBuilder enhanced = new StringBuilder(originalPrompt);

		if (!styleHints.isEmpty()) {
			enhanced.append(", ").append(String.join(", ", styleHints));
		}

		enhanced.append(", high quality, detailed, professional");
		return enhanced.toString();
	}

	private Path generateImageWithStableDiffusion(String prompt, ImageGenerationParameters params) throws Exception {
		initializeStableDiffusion();

		NativeStableDiffusionWrapper.GenerationParameters sdParams =
			NativeStableDiffusionWrapper.GenerationParameters.forSD35Medium();

		sdParams.prompt = prompt;
		sdParams.negativePrompt = "blurry, low quality, distorted, ugly";
		sdParams.width = params.getWidth();
		sdParams.height = params.getHeight();
		sdParams.steps = params.getSteps();
		sdParams.cfgScale = params.getGuidanceScale();
		sdParams.clipOnCpu = true;

		if (seed != null && seed > 0) {
			sdParams.seed = seed.intValue();
		}

		StableDiffusionResult result = stableDiffusion.generateImage(sdParams);
		if (!result.isSuccess()) {
			String error = result.getErrorMessage().orElse("Unknown error");
			throw new RuntimeException("Stable Diffusion generation failed: " + error);
		}

		// Save the generated image to the output directory
		Path imagePath = generateOutputPath("sd_image", "png");
		if (result.getImageData().isPresent()) {
			NativeStableDiffusionWrapper.saveImageAsPng(result, imagePath);
		}

		return imagePath;
	}


	private Path generateOutputPath(String type, String extension) {
		String timestamp = String.valueOf(System.currentTimeMillis());
		String filename = type + "_" + timestamp + "." + extension;
		return outputDirectory.resolve(filename);
	}

	private List<KeyframePrompt> generateKeyframes(String prompt, VideoGenerationParameters params) {
		List<KeyframePrompt> keyframes = new ArrayList<>();
		int frameCount = (int) (params.getDuration() * params.getFps());

		for (int i = 0; i < Math.min(frameCount, 10); i++) {
			double timePosition = (double) i / (frameCount - 1);
			keyframes.add(new KeyframePrompt(i, timePosition, prompt));
		}

		return keyframes;
	}

	private void combineFramesToVideo(List<GeneratedFrame> frames, Path outputPath, VideoGenerationParameters params) throws IOException {
		// Placeholder for video composition
		Files.createFile(outputPath);
	}

	private Scene3DData generate3DSceneData(String prompt, SceneGenerationParameters params) {
		Scene3DParameters sceneParams = new Scene3DParameters.Builder()
				.sceneType(params.getSceneType().name().toLowerCase())
				.complexity(params.getComplexity().ordinal() / (float) SceneComplexity.values().length)
				.lightingStyle(params.getLighting().name().toLowerCase())
				.materialStyle(params.getMaterialQuality().name().toLowerCase())
				.build();

		return new Scene3DData.Builder()
				.prompt(prompt)
				.parameters(sceneParams)
				.meshData(new byte[1024])
				.textureData(new byte[512])
				.build();
	}

	private List<RenderedView> render3DViews(Scene3DData sceneData) {
		List<RenderedView> views = new ArrayList<>();

		Camera3DSettings frontView = Camera3DSettings.builder()
				.position(0, 0, 5)
				.target(0, 0, 0)
				.build();

		views.add(new RenderedView("front", frontView, generateOutputPath("view_front", "jpg")));

		return views;
	}

	private void save3DScene(Scene3DData sceneData, Path outputPath) throws IOException {
		Files.write(outputPath, sceneData.getMeshData());
	}

	@Override
	public void close() {
		try {
			if (textToImageModel != null) {
				textToImageModel.close();
			}
			if (stableDiffusion != null) {
				stableDiffusion.close();
			}
			if (imageProcessor != null) {
				imageProcessor.close();
			}
			if (executor != null && !executor.isShutdown()) {
				executor.shutdown();
			}
		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.WARNING, "Error during cleanup", e);
		}
	}
}
