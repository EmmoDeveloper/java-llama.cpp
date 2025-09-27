package de.kherud.llama.diffusion;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/**
 * Java wrapper for stable-diffusion.cpp executable.
 *
 * This class provides integration with stable-diffusion.cpp to generate images
 * using Stable Diffusion models in GGUF format.
 *
 * Supports SD3.5 Medium models with appropriate text encoders.
 */
public class StableDiffusionCppWrapper {
	private static final System.Logger LOGGER = System.getLogger(StableDiffusionCppWrapper.class.getName());

	private final String executablePath;
	private final String modelPath;
	private final String clipLPath;
	private final String clipGPath;
	private final String t5xxlPath;
	private final Path outputDirectory;

	public static class Builder {
		private String executablePath = "/opt/stable-diffusion.cpp/build/bin/sd";
		private String modelPath;
		private String clipLPath;
		private String clipGPath;
		private String t5xxlPath;
		private Path outputDirectory = Paths.get("./generated_images");

		public Builder executable(String path) {
			this.executablePath = path;
			return this;
		}

		public Builder model(String path) {
			this.modelPath = path;
			return this;
		}

		public Builder clipL(String path) {
			this.clipLPath = path;
			return this;
		}

		public Builder clipG(String path) {
			this.clipGPath = path;
			return this;
		}

		public Builder t5xxl(String path) {
			this.t5xxlPath = path;
			return this;
		}

		public Builder outputDirectory(Path dir) {
			this.outputDirectory = dir;
			return this;
		}

		public StableDiffusionCppWrapper build() {
			if (modelPath == null) {
				throw new IllegalArgumentException("Model path is required");
			}
			return new StableDiffusionCppWrapper(this);
		}
	}

	private StableDiffusionCppWrapper(Builder builder) {
		this.executablePath = builder.executablePath;
		this.modelPath = builder.modelPath;
		this.clipLPath = builder.clipLPath;
		this.clipGPath = builder.clipGPath;
		this.t5xxlPath = builder.t5xxlPath;
		this.outputDirectory = builder.outputDirectory;

		// Ensure output directory exists
		try {
			Files.createDirectories(outputDirectory);
		} catch (IOException e) {
			LOGGER.log(System.Logger.Level.WARNING, "Failed to create output directory: " + outputDirectory, e);
		}
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Parameters for image generation
	 */
	public static class GenerationParams {
		public String prompt = "";
		public String negativePrompt = "";
		public int width = 768;
		public int height = 768;
		public int steps = 30;
		public float cfgScale = 7.0f;
		public float slgScale = 2.5f;  // Good for SD3.5 Medium
		public String samplingMethod = "euler";
		public long seed = -1;
		public boolean verbose = false;
		public boolean clipOnCpu = true;  // Save GPU memory
	}

	/**
	 * Generate an image using stable-diffusion.cpp
	 */
	public GenerationResult generateImage(GenerationParams params) {
		String outputPath = outputDirectory.resolve("sd_" + UUID.randomUUID() + ".png").toString();

		List<String> command = buildCommand(params, outputPath);

		LOGGER.log(System.Logger.Level.INFO, "Executing: " + String.join(" ", command));

		try {
			ProcessBuilder pb = new ProcessBuilder(command);
			pb.redirectErrorStream(true);

			Process process = pb.start();

			// Capture output
			StringBuilder output = new StringBuilder();
			try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
				String line;
				while ((line = reader.readLine()) != null) {
					output.append(line).append("\n");
					if (params.verbose) {
						System.out.println(line);
					}
				}
			}

			boolean finished = process.waitFor(120, TimeUnit.SECONDS);
			if (!finished) {
				process.destroyForcibly();
				return new GenerationResult(false, null, "Generation timed out after 120 seconds");
			}

			int exitCode = process.exitValue();
			if (exitCode == 0) {
				Path generatedImage = Paths.get(outputPath);
				if (Files.exists(generatedImage)) {
					return new GenerationResult(true, generatedImage, "Image generated successfully");
				} else {
					return new GenerationResult(false, null, "Image file not found after generation");
				}
			} else {
				return new GenerationResult(false, null,
					"Generation failed with exit code " + exitCode + "\nOutput: " + output.toString());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Error during image generation", e);
			return new GenerationResult(false, null, "Error: " + e.getMessage());
		}
	}

	private List<String> buildCommand(GenerationParams params, String outputPath) {
		List<String> cmd = new ArrayList<>();

		// Executable
		cmd.add(executablePath);

		// Model paths
		cmd.add("-m");
		cmd.add(modelPath);

		// Add text encoders for SD3.5
		if (clipLPath != null) {
			cmd.add("--clip_l");
			cmd.add(clipLPath);
		}
		if (clipGPath != null) {
			cmd.add("--clip_g");
			cmd.add(clipGPath);
		}
		if (t5xxlPath != null) {
			cmd.add("--t5xxl");
			cmd.add(t5xxlPath);
		}

		// Resolution
		cmd.add("-W");
		cmd.add(String.valueOf(params.width));
		cmd.add("-H");
		cmd.add(String.valueOf(params.height));

		// Prompt
		cmd.add("-p");
		cmd.add(params.prompt);

		// Negative prompt
		if (params.negativePrompt != null && !params.negativePrompt.isEmpty()) {
			cmd.add("-n");
			cmd.add(params.negativePrompt);
		}

		// Generation parameters
		cmd.add("--cfg-scale");
		cmd.add(String.valueOf(params.cfgScale));

		// SLG scale for SD3.5 Medium
		if (params.slgScale > 0) {
			cmd.add("--slg-scale");
			cmd.add(String.valueOf(params.slgScale));
		}

		cmd.add("--steps");
		cmd.add(String.valueOf(params.steps));

		cmd.add("--sampling-method");
		cmd.add(params.samplingMethod);

		// Seed
		if (params.seed > 0) {
			cmd.add("--seed");
			cmd.add(String.valueOf(params.seed));
		}

		// Output
		cmd.add("-o");
		cmd.add(outputPath);

		// Options
		if (params.verbose) {
			cmd.add("-v");
		}

		if (params.clipOnCpu) {
			cmd.add("--clip-on-cpu");
		}

		return cmd;
	}

	public static class GenerationResult {
		public final boolean success;
		public final Path imagePath;
		public final String message;

		public GenerationResult(boolean success, Path imagePath, String message) {
			this.success = success;
			this.imagePath = imagePath;
			this.message = message;
		}
	}

	/**
	 * Check if stable-diffusion.cpp is available
	 */
	public boolean isAvailable() {
		try {
			Process process = new ProcessBuilder(executablePath, "--help")
					.redirectErrorStream(true)
					.start();

			boolean finished = process.waitFor(5, TimeUnit.SECONDS);
			if (finished) {
				return process.exitValue() == 0;
			}
			process.destroyForcibly();
			return false;
		} catch (Exception e) {
			return false;
		}
	}

	/**
	 * Get version information
	 */
	public String getVersion() {
		try {
			Process process = new ProcessBuilder(executablePath, "--version")
					.redirectErrorStream(true)
					.start();

			StringBuilder output = new StringBuilder();
			try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
				String line;
				while ((line = reader.readLine()) != null) {
					output.append(line).append("\n");
				}
			}

			process.waitFor(5, TimeUnit.SECONDS);
			return output.toString();
		} catch (Exception e) {
			return "Version information unavailable: " + e.getMessage();
		}
	}
}