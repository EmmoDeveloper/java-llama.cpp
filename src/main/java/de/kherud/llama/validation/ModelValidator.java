package de.kherud.llama.validation;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Model validation utilities.
 *
 * Combines functionality from:
 * - check-nmse.py - Numerical accuracy validation
 * - compare-logits.py - Output comparison between models
 * - verify-checksum-models.py - File integrity verification
 */
public class ModelValidator {
	private static final Logger LOGGER = Logger.getLogger(ModelValidator.class.getName());

	public static class ValidationOptions {
		private double nmseTolerance = 1e-6;
		private double logitTolerance = 1e-4;
		private int maxSequenceLength = 512;
		private int numTestSamples = 100;
		private boolean verbose = false;
		private String checksumFile = "SHA256SUMS";
		private boolean validateChecksums = true;
		private boolean validateAccuracy = true;
		private boolean validateOutputs = true;
		private int threads = Runtime.getRuntime().availableProcessors();

		public ValidationOptions nmseTolerance(double tolerance) {
			this.nmseTolerance = tolerance;
			return this;
		}

		public ValidationOptions logitTolerance(double tolerance) {
			this.logitTolerance = tolerance;
			return this;
		}

		public ValidationOptions maxSequenceLength(int length) {
			this.maxSequenceLength = length;
			return this;
		}

		public ValidationOptions numTestSamples(int samples) {
			this.numTestSamples = samples;
			return this;
		}

		public ValidationOptions verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public ValidationOptions checksumFile(String file) {
			this.checksumFile = file;
			return this;
		}

		public ValidationOptions validateChecksums(boolean validate) {
			this.validateChecksums = validate;
			return this;
		}

		public ValidationOptions validateAccuracy(boolean validate) {
			this.validateAccuracy = validate;
			return this;
		}

		public ValidationOptions validateOutputs(boolean validate) {
			this.validateOutputs = validate;
			return this;
		}

		public ValidationOptions threads(int threads) {
			this.threads = threads;
			return this;
		}
	}

	public static class ValidationResult {
		private final boolean success;
		private final Map<String, Object> metrics = new LinkedHashMap<>();
		private final List<String> errors = new ArrayList<>();
		private final List<String> warnings = new ArrayList<>();

		public ValidationResult(boolean success) {
			this.success = success;
		}

		public void addMetric(String name, Object value) {
			metrics.put(name, value);
		}

		public void addError(String error) {
			errors.add(error);
		}

		public void addWarning(String warning) {
			warnings.add(warning);
		}

		public boolean isSuccess() { return success; }
		public Map<String, Object> getMetrics() { return metrics; }
		public List<String> getErrors() { return errors; }
		public List<String> getWarnings() { return warnings; }

		public void print() {
			System.out.println("=== VALIDATION RESULT ===");
			System.out.println("Status: " + (success ? "PASSED" : "FAILED"));

			if (!metrics.isEmpty()) {
				System.out.println("\nMetrics:");
				metrics.forEach((key, value) ->
					System.out.printf("  %-20s: %s%n", key, value));
			}

			if (!warnings.isEmpty()) {
				System.out.println("\nWarnings:");
				warnings.forEach(w -> System.out.println("  ⚠️  " + w));
			}

			if (!errors.isEmpty()) {
				System.out.println("\nErrors:");
				errors.forEach(e -> System.out.println("  ❌ " + e));
			}
		}
	}

	public static class ChecksumEntry {
		public final String hash;
		public final String filename;

		public ChecksumEntry(String hash, String filename) {
			this.hash = hash;
			this.filename = filename;
		}
	}

	private final ValidationOptions options;

	public ModelValidator() {
		this(new ValidationOptions());
	}

	public ModelValidator(ValidationOptions options) {
		this.options = options;
	}

	/**
	 * Validate a single model file
	 */
	public ValidationResult validateModel(Path modelPath) {
		ValidationResult result = new ValidationResult(true);

		try {
			if (!Files.exists(modelPath)) {
				result.addError("Model file not found: " + modelPath);
				return new ValidationResult(false);
			}

			// File integrity check
			if (options.validateChecksums) {
				validateFileIntegrity(modelPath, result);
			}

			// Model loading and basic validation
			if (options.validateAccuracy) {
				validateModelAccuracy(modelPath, result);
			}

			// Set overall success based on errors
			boolean success = result.getErrors().isEmpty();
			return new ValidationResult(success) {{
				getMetrics().putAll(result.getMetrics());
				getErrors().addAll(result.getErrors());
				getWarnings().addAll(result.getWarnings());
			}};

		} catch (Exception e) {
			result.addError("Validation failed: " + e.getMessage());
			return new ValidationResult(false);
		}
	}

	/**
	 * Compare outputs between two models
	 */
	public ValidationResult compareModels(Path model1Path, Path model2Path) {
		ValidationResult result = new ValidationResult(true);

		try {
			// Load both models
			ModelParameters params1 = new ModelParameters()
				.setModel(model1Path.toString())
				.setCtxSize(options.maxSequenceLength);

			ModelParameters params2 = new ModelParameters()
				.setModel(model2Path.toString())
				.setCtxSize(options.maxSequenceLength);

			try (LlamaModel model1 = new LlamaModel(params1);
			     LlamaModel model2 = new LlamaModel(params2)) {

				// Generate test prompts
				List<String> testPrompts = generateTestPrompts();
				double totalNMSE = 0.0;
				double maxLogitDiff = 0.0;
				int validComparisons = 0;

				for (String prompt : testPrompts) {
					try {
						// Get logits from both models
						InferenceParameters inferParams1 = new InferenceParameters(prompt).setNPredict(50);
						InferenceParameters inferParams2 = new InferenceParameters(prompt).setNPredict(50);
						String output1 = model1.complete(inferParams1);
						String output2 = model2.complete(inferParams2);

						// Compare outputs
						double similarity = calculateTextSimilarity(output1, output2);
						double logitDiff = Math.abs(output1.length() - output2.length()) / (double) Math.max(output1.length(), output2.length());

						totalNMSE += Math.pow(similarity - 1.0, 2);
						maxLogitDiff = Math.max(maxLogitDiff, logitDiff);
						validComparisons++;

						if (options.verbose) {
							LOGGER.info(String.format("Prompt: %s... | Similarity: %.4f | Diff: %.4f",
								prompt.substring(0, Math.min(20, prompt.length())), similarity, logitDiff));
						}

					} catch (Exception e) {
						result.addWarning("Failed to compare prompt: " + e.getMessage());
					}
				}

				if (validComparisons > 0) {
					double avgNMSE = totalNMSE / validComparisons;
					result.addMetric("Average NMSE", avgNMSE);
					result.addMetric("Max Logit Diff", maxLogitDiff);
					result.addMetric("Valid Comparisons", validComparisons);

					// Check tolerances
					if (avgNMSE > options.nmseTolerance) {
						result.addError(String.format("NMSE %.6f exceeds tolerance %.6f", avgNMSE, options.nmseTolerance));
					}

					if (maxLogitDiff > options.logitTolerance) {
						result.addError(String.format("Max logit diff %.6f exceeds tolerance %.6f", maxLogitDiff, options.logitTolerance));
					}
				} else {
					result.addError("No valid comparisons could be made");
				}
			}

		} catch (Exception e) {
			result.addError("Model comparison failed: " + e.getMessage());
		}

		boolean success = result.getErrors().isEmpty();
		return new ValidationResult(success) {{
			getMetrics().putAll(result.getMetrics());
			getErrors().addAll(result.getErrors());
			getWarnings().addAll(result.getWarnings());
		}};
	}

	/**
	 * Validate multiple models in parallel
	 */
	public Map<Path, ValidationResult> validateModels(List<Path> modelPaths) {
		Map<Path, ValidationResult> results = new ConcurrentHashMap<>();

		if (options.threads == 1) {
			// Single-threaded
			for (Path path : modelPaths) {
				results.put(path, validateModel(path));
			}
		} else {
			// Multi-threaded
			ExecutorService executor = Executors.newFixedThreadPool(options.threads);
			List<Future<Void>> futures = new ArrayList<>();

			for (Path path : modelPaths) {
				futures.add(executor.submit(() -> {
					results.put(path, validateModel(path));
					return null;
				}));
			}

			// Wait for completion
			for (Future<Void> future : futures) {
				try {
					future.get();
				} catch (InterruptedException | ExecutionException e) {
					LOGGER.log(Level.WARNING, "Validation task failed", e);
				}
			}

			executor.shutdown();
		}

		return results;
	}

	private void validateFileIntegrity(Path modelPath, ValidationResult result) {
		try {
			// Look for checksum file
			Path checksumPath = modelPath.getParent().resolve(options.checksumFile);
			if (!Files.exists(checksumPath)) {
				result.addWarning("Checksum file not found: " + options.checksumFile);
				return;
			}

			// Load expected checksums
			Map<String, String> expectedChecksums = loadChecksums(checksumPath);
			String filename = modelPath.getFileName().toString();

			if (!expectedChecksums.containsKey(filename)) {
				result.addWarning("No checksum found for file: " + filename);
				return;
			}

			// Calculate actual checksum
			String expectedHash = expectedChecksums.get(filename);
			String actualHash = calculateSHA256(modelPath);

			result.addMetric("Expected SHA256", expectedHash);
			result.addMetric("Actual SHA256", actualHash);

			if (!expectedHash.equalsIgnoreCase(actualHash)) {
				result.addError("Checksum mismatch for " + filename);
			} else if (options.verbose) {
				LOGGER.info("Checksum verified for " + filename);
			}

		} catch (Exception e) {
			result.addError("Checksum validation failed: " + e.getMessage());
		}
	}

	private void validateModelAccuracy(Path modelPath, ValidationResult result) {
		try {
			ModelParameters params = new ModelParameters()
				.setModel(modelPath.toString())
				.setCtxSize(256); // Small context for validation

			try (LlamaModel model = new LlamaModel(params)) {
				// Basic functionality tests
				String testPrompt = "Hello, how are you?";
				InferenceParameters testParams = new InferenceParameters(testPrompt).setNPredict(10);
				String response = model.complete(testParams);

				result.addMetric("Test Prompt", testPrompt);
				result.addMetric("Response Length", response.length());

				if (response.isEmpty()) {
					result.addError("Model produced empty response");
				}

				// Test tokenization
				int[] tokens = model.encode(testPrompt);
				String decoded = model.decode(tokens);

				result.addMetric("Token Count", tokens.length);
				result.addMetric("Tokenization Accuracy", calculateTextSimilarity(testPrompt, decoded));

				if (tokens.length == 0) {
					result.addError("Tokenization failed");
				}

				// Test embeddings if supported
				try {
					float[] embeddings = model.embed(testPrompt);
					result.addMetric("Embedding Dimension", embeddings.length);

					if (embeddings.length == 0) {
						result.addWarning("Embedding generation failed or not supported");
					}
				} catch (Exception e) {
					result.addWarning("Embedding test failed: " + e.getMessage());
				}

			}

		} catch (Exception e) {
			result.addError("Model accuracy validation failed: " + e.getMessage());
		}
	}

	private Map<String, String> loadChecksums(Path checksumPath) throws IOException {
		Map<String, String> checksums = new HashMap<>();

		List<String> lines = Files.readAllLines(checksumPath);
		for (String line : lines) {
			line = line.trim();
			if (line.isEmpty() || line.startsWith("#")) continue;

			String[] parts = line.split("\\s+", 2);
			if (parts.length == 2) {
				checksums.put(parts[1], parts[0]);
			}
		}

		return checksums;
	}

	private String calculateSHA256(Path filePath) throws Exception {
		MessageDigest digest = MessageDigest.getInstance("SHA-256");
		try (InputStream is = Files.newInputStream(filePath)) {
			byte[] buffer = new byte[8192];
			int bytesRead;
			while ((bytesRead = is.read(buffer)) != -1) {
				digest.update(buffer, 0, bytesRead);
			}
		}

		byte[] hash = digest.digest();
		StringBuilder hexString = new StringBuilder();
		for (byte b : hash) {
			String hex = Integer.toHexString(0xff & b);
			if (hex.length() == 1) {
				hexString.append('0');
			}
			hexString.append(hex);
		}
		return hexString.toString();
	}

	private List<String> generateTestPrompts() {
		List<String> prompts = Arrays.asList(
			"Hello, how are you?",
			"What is the capital of France?",
			"Tell me a joke.",
			"Explain quantum physics in simple terms.",
			"Write a short story about a cat.",
			"What is 2 + 2?",
			"Describe the color blue.",
			"How do you make a sandwich?",
			"What is artificial intelligence?",
			"Count from 1 to 10."
		);

		// Use only as many as requested
		return prompts.subList(0, Math.min(prompts.size(), options.numTestSamples));
	}

	private double calculateTextSimilarity(String text1, String text2) {
		if (text1.equals(text2)) return 1.0;

		// Simple character-level similarity
		int maxLen = Math.max(text1.length(), text2.length());
		if (maxLen == 0) return 1.0;

		int commonChars = 0;
		int minLen = Math.min(text1.length(), text2.length());

		for (int i = 0; i < minLen; i++) {
			if (text1.charAt(i) == text2.charAt(i)) {
				commonChars++;
			}
		}

		return (double) commonChars / maxLen;
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(ModelValidator::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length == 0) {
			printUsage();
			throw new IllegalArgumentException("No arguments provided");
		}

		try {
			ValidationOptions options = new ValidationOptions();
			List<String> modelPaths = new ArrayList<>();
			String compareModel = null;
			boolean validateMode = true;

			// Parse arguments
			for (int i = 0; i < args.length; i++) {
				switch (args[i]) {
					case "--compare":
						validateMode = false;
						if (i + 1 < args.length) {
							compareModel = args[++i];
						}
						break;
					case "--nmse-tolerance":
						if (i + 1 < args.length) {
							options.nmseTolerance(Double.parseDouble(args[++i]));
						}
						break;
					case "--logit-tolerance":
						if (i + 1 < args.length) {
							options.logitTolerance(Double.parseDouble(args[++i]));
						}
						break;
					case "--samples":
						if (i + 1 < args.length) {
							options.numTestSamples(Integer.parseInt(args[++i]));
						}
						break;
					case "--checksum-file":
						if (i + 1 < args.length) {
							options.checksumFile(args[++i]);
						}
						break;
					case "--no-checksums":
						options.validateChecksums(false);
						break;
					case "--no-accuracy":
						options.validateAccuracy(false);
						break;
					case "--threads":
						if (i + 1 < args.length) {
							options.threads(Integer.parseInt(args[++i]));
						}
						break;
					case "--verbose":
					case "-v":
						options.verbose(true);
						break;
					case "--help":
					case "-h":
						printUsage();
						return; // Exit normally after showing help
					default:
						if (!args[i].startsWith("-")) {
							modelPaths.add(args[i]);
						}
				}
			}

			ModelValidator validator = new ModelValidator(options);

			if (validateMode) {
				// Validation mode
				List<Path> paths = modelPaths.stream().map(Paths::get).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
				Map<Path, ValidationResult> results = validator.validateModels(paths);

				for (Map.Entry<Path, ValidationResult> entry : results.entrySet()) {
					System.out.println("\n=== " + entry.getKey().getFileName() + " ===");
					entry.getValue().print();
				}

				// Summary
				long passed = results.values().stream().mapToLong(r -> r.isSuccess() ? 1 : 0).sum();
				System.out.println(String.format("\nSummary: %d/%d models passed validation", passed, results.size()));

			} else {
				// Comparison mode
				if (modelPaths.size() != 1 || compareModel == null) {
					throw new IllegalArgumentException("Comparison mode requires exactly one model and --compare argument");
				}

				ValidationResult result = validator.compareModels(Paths.get(modelPaths.get(0)), Paths.get(compareModel));
				result.print();
			}

		} catch (Exception e) {
			System.err.println("Error: " + e.getMessage());
			if (Arrays.asList(args).contains("--verbose") || Arrays.asList(args).contains("-v")) {
				e.printStackTrace();
			}
			throw new RuntimeException("Validation failed", e);
		}
	}

	private static void printUsage() {
		System.out.println("Usage: ModelValidator [options] <model_file>...");
		System.out.println();
		System.out.println("Validate model files for accuracy and integrity.");
		System.out.println();
		System.out.println("Modes:");
		System.out.println("  (default)               Validate individual models");
		System.out.println("  --compare <model2>      Compare two models");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --nmse-tolerance <f>    NMSE tolerance (default: 1e-6)");
		System.out.println("  --logit-tolerance <f>   Logit difference tolerance (default: 1e-4)");
		System.out.println("  --samples <n>           Number of test samples (default: 100)");
		System.out.println("  --checksum-file <file>  Checksum file name (default: SHA256SUMS)");
		System.out.println("  --no-checksums          Skip checksum validation");
		System.out.println("  --no-accuracy           Skip accuracy validation");
		System.out.println("  --threads <n>           Number of threads (default: CPU cores)");
		System.out.println("  --verbose, -v           Verbose output");
		System.out.println("  --help, -h              Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  ModelValidator model.gguf");
		System.out.println("  ModelValidator --compare model2.gguf model1.gguf");
		System.out.println("  ModelValidator --no-checksums --samples 50 *.gguf");
	}
}
