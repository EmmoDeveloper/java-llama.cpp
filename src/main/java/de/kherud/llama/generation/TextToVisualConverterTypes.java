package de.kherud.llama.generation;

import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

/**
 * Supporting types and data structures for TextToVisualConverter.
 * Contains enums, data classes, builders, and result types.
 */
public class TextToVisualConverterTypes {

	// Enums
	public enum OutputFormat {
		HIGH_RESOLUTION_IMAGE,
		MEDIUM_RESOLUTION_IMAGE,
		LOW_RESOLUTION_IMAGE,
		SCENE_3D,
		VIDEO_2D,
		VIDEO_3D,
		ANIMATION,
		ENHANCED_IMAGE,
		STYLE_TRANSFER
	}

	public enum ImageQuality {
		DRAFT, STANDARD, HIGH, ULTRA
	}

	public enum VideoQuality {
		DRAFT, STANDARD, HIGH, ULTRA
	}

	public enum VideoFormat {
		MP4, AVI, MOV, WEBM
	}

	public enum SceneType {
		LANDSCAPE, ARCHITECTURAL, CHARACTER, OBJECT
	}

	public enum SceneComplexity {
		LOW, MEDIUM, HIGH, ULTRA
	}

	public enum LightingStyle {
		NATURAL, DRAMATIC, AMBIENT, STUDIO
	}

	public enum MaterialQuality {
		LOW, STANDARD, HIGH, ULTRA
	}

	public enum SceneFormat {
		GLB, GLTF, OBJ, FBX
	}

	public enum GenerationQuality {
		DRAFT(10, 1.0f, 256),
		STANDARD(20, 5.0f, 512),
		HIGH(50, 7.5f, 1024),
		ULTRA(100, 10.0f, 2048);

		private final int steps;
		private final float guidanceScale;
		private final int maxResolution;

		GenerationQuality(int steps, float guidanceScale, int maxResolution) {
			this.steps = steps;
			this.guidanceScale = guidanceScale;
			this.maxResolution = maxResolution;
		}

		public int getSteps() { return steps; }
		public float getGuidanceScale() { return guidanceScale; }
		public int getMaxResolution() { return maxResolution; }
	}

	public enum PromptEngineeringStrategy {
		DIRECT,          // Use prompt as-is
		ENHANCED,        // AI-enhanced prompt
		STYLE_FOCUSED,   // Optimize for specific style
		DETAIL_RICH,     // Add detailed descriptions
		COMPOSITION_AWARE // Optimize for good composition
	}

	public enum NoiseScheduler {
		DDPM,           // Denoising Diffusion Probabilistic Models
		DDIM,           // Denoising Diffusion Implicit Models
		EULER_A,        // Euler Ancestral
		DPM_PLUS_PLUS,  // DPM++ Scheduler
		KARRAS          // Karras Scheduler
	}

	public enum SamplingMethod {
		EULER,
		HEUN,
		DPM2,
		DPM2_A,
		DPMPP_2M,
		DPMPP_SDE,
		DPMPP_2M_KARRAS
	}

	// Data classes
	public static class Vector3D {
		public final float x, y, z;

		public Vector3D(float x, float y, float z) {
			this.x = x;
			this.y = y;
			this.z = z;
		}

		public static Vector3D origin() { return new Vector3D(0, 0, 0); }
		public static Vector3D forward() { return new Vector3D(0, 0, -1); }
		public static Vector3D up() { return new Vector3D(0, 1, 0); }
	}

	public static class Camera3DSettings {
		private final Vector3D position;
		private final Vector3D target;
		private final float fieldOfView;
		private final float nearPlane;
		private final float farPlane;

		public Camera3DSettings(Vector3D position, Vector3D target, float fieldOfView, float nearPlane, float farPlane) {
			this.position = position;
			this.target = target;
			this.fieldOfView = fieldOfView;
			this.nearPlane = nearPlane;
			this.farPlane = farPlane;
		}

		public Vector3D getPosition() { return position; }
		public Vector3D getTarget() { return target; }
		public float getFieldOfView() { return fieldOfView; }
		public float getNearPlane() { return nearPlane; }
		public float getFarPlane() { return farPlane; }

		public static Builder builder() { return new Builder(); }

		public static class Builder {
			private Vector3D position = new Vector3D(0, 0, 5);
			private Vector3D target = Vector3D.origin();
			private float fieldOfView = 60.0f;
			private float nearPlane = 0.1f;
			private float farPlane = 1000.0f;

			public Builder position(float x, float y, float z) {
				this.position = new Vector3D(x, y, z);
				return this;
			}

			public Builder target(float x, float y, float z) {
				this.target = new Vector3D(x, y, z);
				return this;
			}

			public Builder fov(float fov) {
				this.fieldOfView = fov;
				return this;
			}

			public Builder nearFar(float near, float far) {
				this.nearPlane = near;
				this.farPlane = far;
				return this;
			}

			public Camera3DSettings build() {
				return new Camera3DSettings(position, target, fieldOfView, nearPlane, farPlane);
			}
		}
	}

	public static class ImageGenerationParameters {
		private final int width;
		private final int height;
		private final int steps;
		private final float guidanceScale;
		private final ImageQuality quality;
		private final String stylePreset;
		private final List<String> styleHints;
		private final String negativePrompt;
		private final long seed;
		private final float creativityLevel;

		private ImageGenerationParameters(Builder builder) {
			this.width = builder.width;
			this.height = builder.height;
			this.steps = builder.steps;
			this.guidanceScale = builder.guidanceScale;
			this.quality = builder.quality;
			this.stylePreset = builder.stylePreset;
			this.styleHints = List.copyOf(builder.styleHints);
			this.negativePrompt = builder.negativePrompt;
			this.seed = builder.seed;
			this.creativityLevel = builder.creativityLevel;
		}

		public static ImageGenerationParameters defaultParams() {
			return new Builder().build();
		}

		public static ImageGenerationParameters forVideo(int width, int height) {
			return new Builder()
				.width(width)
				.height(height)
				.steps(30)
				.guidanceScale(5.0f)
				.build();
		}

		public int getWidth() { return width; }
		public int getHeight() { return height; }
		public int getSteps() { return steps; }
		public float getGuidanceScale() { return guidanceScale; }
		public ImageQuality getQuality() { return quality; }
		public String getStylePreset() { return stylePreset; }
		public List<String> getStyleHints() { return styleHints; }
		public String getNegativePrompt() { return negativePrompt; }
		public long getSeed() { return seed; }
		public float getCreativityLevel() { return creativityLevel; }

		public static class Builder {
			private int width = 512;
			private int height = 512;
			private int steps = 50;
			private float guidanceScale = 7.5f;
			private ImageQuality quality = ImageQuality.STANDARD;
			private String stylePreset = null;
			private List<String> styleHints = new ArrayList<>();
			private String negativePrompt = null;
			private long seed = -1;
			private float creativityLevel = 0.7f;

			public Builder width(int width) {
				if (width <= 0) throw new IllegalArgumentException("Width must be positive");
				this.width = width;
				return this;
			}

			public Builder height(int height) {
				if (height <= 0) throw new IllegalArgumentException("Height must be positive");
				this.height = height;
				return this;
			}

			public Builder resolution(int width, int height) {
				return width(width).height(height);
			}

			public Builder steps(int steps) {
				this.steps = steps;
				return this;
			}

			public Builder guidanceScale(float scale) {
				this.guidanceScale = scale;
				return this;
			}

			public Builder quality(ImageQuality quality) {
				this.quality = quality;
				return this;
			}

			public Builder stylePreset(String preset) {
				this.stylePreset = preset;
				return this;
			}

			public Builder styleHints(List<String> hints) {
				this.styleHints = new ArrayList<>(hints);
				return this;
			}

			public Builder addStyleHint(String hint) {
				this.styleHints.add(hint);
				return this;
			}

			public Builder negativePrompt(String prompt) {
				this.negativePrompt = prompt;
				return this;
			}

			public Builder seed(long seed) {
				this.seed = seed;
				return this;
			}

			public Builder creativity(float level) {
				this.creativityLevel = Math.max(0.0f, Math.min(1.0f, level));
				return this;
			}

			public ImageGenerationParameters build() {
				return new ImageGenerationParameters(this);
			}
		}
	}

	public static class VideoGenerationParameters {
		private final int width;
		private final int height;
		private final float duration;
		private final int fps;
		private final VideoQuality quality;
		private final VideoFormat format;

		private VideoGenerationParameters(Builder builder) {
			this.width = builder.width;
			this.height = builder.height;
			this.duration = builder.duration;
			this.fps = builder.fps;
			this.quality = builder.quality;
			this.format = builder.format;
		}

		public int getWidth() { return width; }
		public int getHeight() { return height; }
		public float getDuration() { return duration; }
		public int getFps() { return fps; }
		public VideoQuality getQuality() { return quality; }
		public VideoFormat getFormat() { return format; }

		public static class Builder {
			private int width = 640;
			private int height = 480;
			private float duration = 5.0f;
			private int fps = 24;
			private VideoQuality quality = VideoQuality.STANDARD;
			private VideoFormat format = VideoFormat.MP4;

			public Builder width(int width) {
				this.width = width;
				return this;
			}

			public Builder height(int height) {
				this.height = height;
				return this;
			}

			public Builder duration(float duration) {
				if (duration <= 0) throw new IllegalArgumentException("Duration must be positive");
				this.duration = duration;
				return this;
			}

			public Builder fps(int fps) {
				if (fps <= 0) throw new IllegalArgumentException("FPS must be positive");
				this.fps = fps;
				return this;
			}

			public Builder quality(VideoQuality quality) {
				this.quality = quality;
				return this;
			}

			public Builder format(VideoFormat format) {
				this.format = format;
				return this;
			}

			public VideoGenerationParameters build() {
				return new VideoGenerationParameters(this);
			}
		}
	}

	public static class SceneGenerationParameters {
		private final SceneType sceneType;
		private final SceneComplexity complexity;
		private final LightingStyle lighting;
		private final MaterialQuality materialQuality;
		private final SceneFormat outputFormat;

		private SceneGenerationParameters(Builder builder) {
			this.sceneType = builder.sceneType;
			this.complexity = builder.complexity;
			this.lighting = builder.lighting;
			this.materialQuality = builder.materialQuality;
			this.outputFormat = builder.outputFormat;
		}

		public SceneType getSceneType() { return sceneType; }
		public SceneComplexity getComplexity() { return complexity; }
		public LightingStyle getLighting() { return lighting; }
		public MaterialQuality getMaterialQuality() { return materialQuality; }
		public SceneFormat getOutputFormat() { return outputFormat; }

		public static class Builder {
			private SceneType sceneType = SceneType.LANDSCAPE;
			private SceneComplexity complexity = SceneComplexity.MEDIUM;
			private LightingStyle lighting = LightingStyle.NATURAL;
			private MaterialQuality materialQuality = MaterialQuality.STANDARD;
			private SceneFormat outputFormat = SceneFormat.GLB;

			public Builder sceneType(SceneType type) {
				this.sceneType = type;
				return this;
			}

			public Builder complexity(SceneComplexity complexity) {
				this.complexity = complexity;
				return this;
			}

			public Builder lighting(LightingStyle lighting) {
				this.lighting = lighting;
				return this;
			}

			public Builder materialQuality(MaterialQuality quality) {
				this.materialQuality = quality;
				return this;
			}

			public Builder outputFormat(SceneFormat format) {
				this.outputFormat = format;
				return this;
			}

			public SceneGenerationParameters build() {
				return new SceneGenerationParameters(this);
			}
		}
	}

	public static class Scene3DParameters {
		private final String sceneType;
		private final float complexity;
		private final int detailLevel;
		private final String lightingStyle;
		private final String materialStyle;

		private Scene3DParameters(Builder builder) {
			this.sceneType = builder.sceneType;
			this.complexity = builder.complexity;
			this.detailLevel = builder.detailLevel;
			this.lightingStyle = builder.lightingStyle;
			this.materialStyle = builder.materialStyle;
		}

		public static Scene3DParameters defaultParameters() {
			return new Builder().build();
		}

		public String getSceneType() { return sceneType; }
		public float getComplexity() { return complexity; }
		public int getDetailLevel() { return detailLevel; }
		public String getLightingStyle() { return lightingStyle; }
		public String getMaterialStyle() { return materialStyle; }

		public static class Builder {
			private String sceneType = "general";
			private float complexity = 0.7f;
			private int detailLevel = 5;
			private String lightingStyle = "natural";
			private String materialStyle = "realistic";

			public Builder sceneType(String type) {
				this.sceneType = type;
				return this;
			}

			public Builder complexity(float complexity) {
				this.complexity = Math.max(0.0f, Math.min(1.0f, complexity));
				return this;
			}

			public Builder detailLevel(int level) {
				this.detailLevel = Math.max(1, Math.min(10, level));
				return this;
			}

			public Builder lightingStyle(String style) {
				this.lightingStyle = style;
				return this;
			}

			public Builder materialStyle(String style) {
				this.materialStyle = style;
				return this;
			}

			public Scene3DParameters build() {
				return new Scene3DParameters(this);
			}
		}
	}

	public static class GenerationProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;
		private final Map<String, Object> metadata;

		public GenerationProgress(String message, double progress) {
			this(message, progress, Map.of());
		}

		public GenerationProgress(String message, double progress, Map<String, Object> metadata) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
			this.metadata = Map.copyOf(metadata);
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
		public Map<String, Object> getMetadata() { return metadata; }
	}

	public static class KeyframePrompt {
		private final int frameNumber;
		private final double timePosition;
		private final String prompt;

		public KeyframePrompt(int frameNumber, double timePosition, String prompt) {
			this.frameNumber = frameNumber;
			this.timePosition = timePosition;
			this.prompt = prompt;
		}

		public int getFrameNumber() { return frameNumber; }
		public double getTimePosition() { return timePosition; }
		public String getPrompt() { return prompt; }
	}

	public static class GeneratedFrame {
		private final int frameNumber;
		private final double timeProgress;
		private final Path imagePath;
		private final String prompt;

		public GeneratedFrame(int frameNumber, double timeProgress, Path imagePath, String prompt) {
			this.frameNumber = frameNumber;
			this.timeProgress = timeProgress;
			this.imagePath = imagePath;
			this.prompt = prompt;
		}

		public int getFrameNumber() { return frameNumber; }
		public double getTimeProgress() { return timeProgress; }
		public Path getImagePath() { return imagePath; }
		public String getPrompt() { return prompt; }
	}

	public static class Scene3DData {
		private final String prompt;
		private final Scene3DParameters parameters;
		private final byte[] meshData;
		private final byte[] textureData;
		private final Map<String, Object> metadata;

		private Scene3DData(Builder builder) {
			this.prompt = builder.prompt;
			this.parameters = builder.parameters;
			this.meshData = builder.meshData;
			this.textureData = builder.textureData;
			this.metadata = Map.copyOf(builder.metadata);
		}

		public String getPrompt() { return prompt; }
		public Scene3DParameters getParameters() { return parameters; }
		public byte[] getMeshData() { return meshData; }
		public byte[] getTextureData() { return textureData; }
		public Map<String, Object> getMetadata() { return metadata; }

		public static class Builder {
			private String prompt;
			private Scene3DParameters parameters;
			private byte[] meshData = new byte[0];
			private byte[] textureData = new byte[0];
			private Map<String, Object> metadata = new HashMap<>();

			public Builder prompt(String prompt) {
				this.prompt = prompt;
				return this;
			}

			public Builder parameters(Scene3DParameters parameters) {
				this.parameters = parameters;
				return this;
			}

			public Builder meshData(byte[] data) {
				this.meshData = data;
				return this;
			}

			public Builder textureData(byte[] data) {
				this.textureData = data;
				return this;
			}

			public Builder addMetadata(String key, Object value) {
				this.metadata.put(key, value);
				return this;
			}

			public Scene3DData build() {
				return new Scene3DData(this);
			}
		}
	}

	public static class RenderedView {
		private final String viewName;
		private final Camera3DSettings camera;
		private final Path imagePath;

		public RenderedView(String viewName, Camera3DSettings camera, Path imagePath) {
			this.viewName = viewName;
			this.camera = camera;
			this.imagePath = imagePath;
		}

		public String getViewName() { return viewName; }
		public Camera3DSettings getCamera() { return camera; }
		public Path getImagePath() { return imagePath; }
	}

	// Result classes
	public static class GenerationResult {
		private final boolean success;
		private final String message;
		private final String originalPrompt;
		private final String optimizedPrompt;
		private final Path outputPath;
		private final OutputFormat outputFormat;
		private final int width;
		private final int height;
		private final int generationSteps;
		private final float guidanceScale;
		private final Duration duration;
		private final Exception error;
		private final Map<String, Object> metadata;

		private GenerationResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.originalPrompt = builder.originalPrompt;
			this.optimizedPrompt = builder.optimizedPrompt;
			this.outputPath = builder.outputPath;
			this.outputFormat = builder.outputFormat;
			this.width = builder.width;
			this.height = builder.height;
			this.generationSteps = builder.generationSteps;
			this.guidanceScale = builder.guidanceScale;
			this.duration = builder.duration;
			this.error = builder.error;
			this.metadata = Map.copyOf(builder.metadata);
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public String getOriginalPrompt() { return originalPrompt; }
		public String getOptimizedPrompt() { return optimizedPrompt; }
		public Optional<Path> getOutputPath() { return Optional.ofNullable(outputPath); }
		public OutputFormat getOutputFormat() { return outputFormat; }
		public int getWidth() { return width; }
		public int getHeight() { return height; }
		public int getGenerationSteps() { return generationSteps; }
		public float getGuidanceScale() { return guidanceScale; }
		public Duration getDuration() { return duration; }
		public long getGenerationTimeMs() { return duration != null ? duration.toMillis() : 0L; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }
		public Optional<String> getErrorMessage() { return Optional.ofNullable(error).map(Exception::getMessage); }
		public Map<String, Object> getMetadata() { return metadata; }

		public static class Builder {
			private boolean success;
			private String message;
			private String originalPrompt;
			private String optimizedPrompt;
			private Path outputPath;
			private OutputFormat outputFormat;
			private int width;
			private int height;
			private int generationSteps;
			private float guidanceScale;
			private Duration duration = Duration.ZERO;
			private Exception error;
			private Map<String, Object> metadata = new HashMap<>();

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder originalPrompt(String prompt) { this.originalPrompt = prompt; return this; }
			public Builder optimizedPrompt(String prompt) { this.optimizedPrompt = prompt; return this; }
			public Builder outputPath(Path path) { this.outputPath = path; return this; }
			public Builder outputFormat(OutputFormat format) { this.outputFormat = format; return this; }
			public Builder width(int width) { this.width = width; return this; }
			public Builder height(int height) { this.height = height; return this; }
			public Builder generationSteps(int steps) { this.generationSteps = steps; return this; }
			public Builder guidanceScale(float scale) { this.guidanceScale = scale; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder generationTimeMs(long timeMs) { this.duration = Duration.ofMillis(timeMs); return this; }
			public Builder error(Exception error) { this.error = error; return this; }
			public Builder addMetadata(String key, Object value) { this.metadata.put(key, value); return this; }

			public GenerationResult build() { return new GenerationResult(this); }
		}
	}

	public static class Scene3DResult {
		private final boolean success;
		private final String message;
		private final String prompt;
		private final Scene3DData sceneData;
		private final List<RenderedView> renderedViews;
		private final Camera3DSettings cameraSettings;
		private final Duration duration;
		private final Exception error;

		private Scene3DResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.prompt = builder.prompt;
			this.sceneData = builder.sceneData;
			this.renderedViews = List.copyOf(builder.renderedViews);
			this.cameraSettings = builder.cameraSettings;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public String getPrompt() { return prompt; }
		public Optional<Scene3DData> getSceneData() { return Optional.ofNullable(sceneData); }
		public List<RenderedView> getRenderedViews() { return renderedViews; }
		public Camera3DSettings getCameraSettings() { return cameraSettings; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private String prompt;
			private Scene3DData sceneData;
			private List<RenderedView> renderedViews = new ArrayList<>();
			private Camera3DSettings cameraSettings;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder prompt(String prompt) { this.prompt = prompt; return this; }
			public Builder sceneData(Scene3DData data) { this.sceneData = data; return this; }
			public Builder renderedViews(List<RenderedView> views) { this.renderedViews = views; return this; }
			public Builder cameraSettings(Camera3DSettings settings) { this.cameraSettings = settings; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public Scene3DResult build() { return new Scene3DResult(this); }
		}
	}

	public static class VideoGenerationResult {
		private final boolean success;
		private final String message;
		private final String prompt;
		private final Path videoPath;
		private final List<GeneratedFrame> frames;
		private final int fps;
		private final Duration videoDuration;
		private final int totalFrames;
		private final Duration generationDuration;
		private final Exception error;

		private VideoGenerationResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.prompt = builder.prompt;
			this.videoPath = builder.videoPath;
			this.frames = List.copyOf(builder.frames);
			this.fps = builder.fps;
			this.videoDuration = builder.videoDuration;
			this.totalFrames = builder.totalFrames;
			this.generationDuration = builder.generationDuration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public String getPrompt() { return prompt; }
		public Optional<Path> getVideoPath() { return Optional.ofNullable(videoPath); }
		public List<GeneratedFrame> getFrames() { return frames; }
		public int getFps() { return fps; }
		public Duration getVideoDuration() { return videoDuration; }
		public int getTotalFrames() { return totalFrames; }
		public Duration getGenerationDuration() { return generationDuration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private String prompt;
			private Path videoPath;
			private List<GeneratedFrame> frames = new ArrayList<>();
			private int fps;
			private Duration videoDuration = Duration.ZERO;
			private int totalFrames;
			private Duration generationDuration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder prompt(String prompt) { this.prompt = prompt; return this; }
			public Builder videoPath(Path path) { this.videoPath = path; return this; }
			public Builder frames(List<GeneratedFrame> frames) { this.frames = frames; return this; }
			public Builder fps(int fps) { this.fps = fps; return this; }
			public Builder videoDuration(Duration duration) { this.videoDuration = duration; return this; }
			public Builder totalFrames(int frames) { this.totalFrames = frames; return this; }
			public Builder duration(Duration duration) { this.generationDuration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public VideoGenerationResult build() { return new VideoGenerationResult(this); }
		}
	}

	public static class BatchGenerationResult {
		private final boolean success;
		private final String message;
		private final List<GenerationResult> results;
		private final List<String> failedPrompts;
		private final int totalPrompts;
		private final int successfulGenerations;
		private final int failedGenerations;
		private final long totalTimeMs;
		private final Duration duration;
		private final Exception error;

		private BatchGenerationResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.results = List.copyOf(builder.results);
			this.failedPrompts = List.copyOf(builder.failedPrompts);
			this.totalPrompts = builder.totalPrompts;
			this.successfulGenerations = builder.successfulGenerations;
			this.failedGenerations = builder.failedGenerations;
			this.totalTimeMs = builder.totalTimeMs;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public List<GenerationResult> getResults() { return results; }
		public List<String> getFailedPrompts() { return failedPrompts; }
		public int getTotalPrompts() { return totalPrompts; }
		public int getSuccessfulGenerations() { return successfulGenerations; }
		public int getFailedGenerations() { return failedGenerations; }
		public long getTotalTimeMs() { return totalTimeMs; }
		public int getTotalImages() { return totalPrompts; }
		public int getSuccessfulImages() { return successfulGenerations; }
		public int getFailedImages() { return failedGenerations; }
		public double getSuccessRate() { return totalPrompts > 0 ? (double) successfulGenerations / totalPrompts : 0.0; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private List<GenerationResult> results = new ArrayList<>();
			private List<String> failedPrompts = new ArrayList<>();
			private int totalPrompts;
			private int successfulGenerations;
			private int failedGenerations;
			private long totalTimeMs;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder results(List<GenerationResult> results) {
				this.results = results;
				this.successfulGenerations = (int) results.stream().filter(GenerationResult::isSuccess).count();
				this.failedGenerations = results.size() - this.successfulGenerations;
				return this;
			}
			public Builder failedPrompts(List<String> prompts) { this.failedPrompts = prompts; return this; }
			public Builder totalPrompts(int total) { this.totalPrompts = total; return this; }
			public Builder successfulGenerations(int count) { this.successfulGenerations = count; return this; }
			public Builder failedGenerations(int count) { this.failedGenerations = count; return this; }
			public Builder totalTimeMs(long timeMs) { this.totalTimeMs = timeMs; return this; }
			public Builder duration(Duration duration) {
				this.duration = duration;
				this.totalTimeMs = duration.toMillis();
				return this;
			}
			public Builder error(Exception error) { this.error = error; return this; }

			public BatchGenerationResult build() { return new BatchGenerationResult(this); }
		}
	}

	// Builder for main class
	public static class Builder {
		public String textToImageModelPath;
		public String outputDirectory = "./generated_content";
		public Long seed;
		public int batchSize = 8;
		public ExecutorService executor;

		public Builder modelPath(String modelPath) {
			this.textToImageModelPath = modelPath;
			return this;
		}

		public Builder outputDirectory(Path directory) {
			this.outputDirectory = directory.toString();
			return this;
		}

		public Builder outputDirectory(String directory) {
			this.outputDirectory = directory;
			return this;
		}

		public Builder seed(long seed) {
			this.seed = seed;
			return this;
		}

		public Builder batchSize(int batchSize) {
			this.batchSize = batchSize;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public TextToVisualConverter build() {
			return new TextToVisualConverter(this);
		}
	}
}