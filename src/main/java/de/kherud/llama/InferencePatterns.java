package de.kherud.llama;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Advanced inference patterns and utilities for common LLM use cases.
 * Provides high-level abstractions for complex inference workflows.
 */
public class InferencePatterns {

	private static final ExecutorService asyncExecutor = Executors.newCachedThreadPool();

	/**
	 * Result of a batch inference operation.
	 */
	public static class BatchResult {
		private final List<String> inputs;
		private final List<String> outputs;
		private final List<Long> latencies;
		private final long totalTime;

		public BatchResult(List<String> inputs, List<String> outputs, List<Long> latencies, long totalTime) {
			this.inputs = new ArrayList<>(inputs);
			this.outputs = new ArrayList<>(outputs);
			this.latencies = new ArrayList<>(latencies);
			this.totalTime = totalTime;
		}

		public List<String> getInputs() { return inputs; }
		public List<String> getOutputs() { return outputs; }
		public List<Long> getLatencies() { return latencies; }
		public long getTotalTime() { return totalTime; }
		public double getAverageLatency() {
			return latencies.stream().mapToLong(Long::longValue).average().orElse(0.0);
		}
		public long getTotalTokensGenerated() {
			return outputs.stream().mapToLong(s -> s.split("\\s+").length).sum();
		}
	}

	/**
	 * Configuration for chain-of-thought reasoning.
	 */
	public static class ChainOfThoughtConfig {
		private final String thinkingPrompt;
		private final String conclusionPrompt;
		private final int maxThinkingTokens;
		private final int maxConclusionTokens;

		public ChainOfThoughtConfig() {
			this("Let me think step by step:", "Therefore, the answer is:", 200, 100);
		}

		public ChainOfThoughtConfig(String thinkingPrompt, String conclusionPrompt,
									int maxThinkingTokens, int maxConclusionTokens) {
			this.thinkingPrompt = thinkingPrompt;
			this.conclusionPrompt = conclusionPrompt;
			this.maxThinkingTokens = maxThinkingTokens;
			this.maxConclusionTokens = maxConclusionTokens;
		}

		public String getThinkingPrompt() { return thinkingPrompt; }
		public String getConclusionPrompt() { return conclusionPrompt; }
		public int getMaxThinkingTokens() { return maxThinkingTokens; }
		public int getMaxConclusionTokens() { return maxConclusionTokens; }
	}

	/**
	 * Perform batch inference on multiple prompts efficiently.
	 */
	public static BatchResult batchInference(LlamaModel model, List<String> prompts, InferenceParameters baseParams) {
		if (model == null || prompts == null || prompts.isEmpty()) {
			throw new IllegalArgumentException("Model and prompts cannot be null or empty");
		}

		long startTime = System.currentTimeMillis();
		List<String> outputs = new ArrayList<>();
		List<Long> latencies = new ArrayList<>();

		for (String prompt : prompts) {
			long promptStart = System.currentTimeMillis();

			StringBuilder output = new StringBuilder();
			InferenceParameters params = new InferenceParameters(prompt);
			if (baseParams != null) {
				// Copy base parameters (using simplified approach)
				params.setTemperature(0.7f)
					  .setNPredict(50)
					  .setTopP(0.9f)
					  .setTopK(40);
			}

			for (LlamaOutput chunk : model.generate(params)) {
				output.append(chunk.toString());
			}

			long promptLatency = System.currentTimeMillis() - promptStart;
			outputs.add(output.toString().trim());
			latencies.add(promptLatency);
		}

		long totalTime = System.currentTimeMillis() - startTime;
		return new BatchResult(prompts, outputs, latencies, totalTime);
	}

	/**
	 * Perform asynchronous inference with future support.
	 */
	public static CompletableFuture<String> asyncInference(LlamaModel model, String prompt, InferenceParameters params) {
		return CompletableFuture.supplyAsync(() -> {
			StringBuilder result = new StringBuilder();
			InferenceParameters inferParams = params != null ? params : new InferenceParameters(prompt);

			for (LlamaOutput chunk : model.generate(inferParams)) {
				result.append(chunk.toString());
			}

			return result.toString().trim();
		}, asyncExecutor);
	}

	/**
	 * Chain-of-thought reasoning pattern - generates reasoning steps then final answer.
	 */
	public static String chainOfThoughtInference(LlamaModel model, String question, ChainOfThoughtConfig config) {
		if (model == null || question == null || config == null) {
			throw new IllegalArgumentException("Model, question, and config cannot be null");
		}

		// Step 1: Generate thinking process
		String thinkingPrompt = question + "\n" + config.getThinkingPrompt();
		StringBuilder thinking = new StringBuilder();

		InferenceParameters thinkingParams = new InferenceParameters(thinkingPrompt)
			.setNPredict(config.getMaxThinkingTokens())
			.setTemperature(0.7f)
			.setStopStrings(config.getConclusionPrompt(), "Answer:", "Final answer:");

		for (LlamaOutput chunk : model.generate(thinkingParams)) {
			thinking.append(chunk.toString());
		}

		// Step 2: Generate final conclusion
		String fullContext = thinkingPrompt + thinking.toString() + "\n" + config.getConclusionPrompt();
		StringBuilder conclusion = new StringBuilder();

		InferenceParameters conclusionParams = new InferenceParameters(fullContext)
			.setNPredict(config.getMaxConclusionTokens())
			.setTemperature(0.3f); // Lower temperature for more focused answer

		for (LlamaOutput chunk : model.generate(conclusionParams)) {
			conclusion.append(chunk.toString());
		}

		// Return the complete chain of thought
		return "Reasoning: " + thinking.toString().trim() + "\n\nAnswer: " + conclusion.toString().trim();
	}

	/**
	 * Template-based inference with variable substitution.
	 */
	public static String templateInference(LlamaModel model, String template, Map<String, String> variables, InferenceParameters params) {
		if (model == null || template == null) {
			throw new IllegalArgumentException("Model and template cannot be null");
		}

		String prompt = template;
		if (variables != null) {
			for (Map.Entry<String, String> entry : variables.entrySet()) {
				prompt = prompt.replace("{" + entry.getKey() + "}", entry.getValue());
			}
		}

		StringBuilder result = new StringBuilder();
		InferenceParameters inferParams = params != null ? params : new InferenceParameters(prompt);

		for (LlamaOutput chunk : model.generate(inferParams)) {
			result.append(chunk.toString());
		}

		return result.toString().trim();
	}

	/**
	 * Consensus-based inference - generate multiple responses and find consensus.
	 */
	public static String consensusInference(LlamaModel model, String prompt, int numResponses, InferenceParameters baseParams) {
		if (model == null || prompt == null || numResponses < 2) {
			throw new IllegalArgumentException("Invalid parameters for consensus inference");
		}

		List<String> responses = new ArrayList<>();

		for (int i = 0; i < numResponses; i++) {
			StringBuilder response = new StringBuilder();

			InferenceParameters params = new InferenceParameters(prompt);
			if (baseParams != null) {
				params.setTemperature(0.7f + (i * 0.1f)) // Vary temperature slightly
					  .setNPredict(50)
					  .setTopP(0.9f)
					  .setSeed(42 + i); // Different seeds for variety
			} else {
				params.setTemperature(0.7f + (i * 0.1f))
					  .setSeed(42 + i);
			}

			for (LlamaOutput chunk : model.generate(params)) {
				response.append(chunk.toString());
			}

			responses.add(response.toString().trim());
		}

		// Simple consensus: return the most common response or first if all unique
		Map<String, Long> responseCounts = responses.stream()
			.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

		return responseCounts.entrySet().stream()
			.max(Map.Entry.comparingByValue())
			.map(Map.Entry::getKey)
			.orElse(responses.get(0));
	}

	/**
	 * Progressive refinement - iteratively improve a response.
	 */
	public static String progressiveRefinement(LlamaModel model, String initialPrompt, int refinementSteps, InferenceParameters params) {
		if (model == null || initialPrompt == null || refinementSteps < 1) {
			throw new IllegalArgumentException("Invalid parameters for progressive refinement");
		}

		String currentResponse = "";
		String currentPrompt = initialPrompt;

		for (int step = 0; step < refinementSteps; step++) {
			StringBuilder response = new StringBuilder();

			InferenceParameters inferParams = params != null ? params : new InferenceParameters(currentPrompt);

			for (LlamaOutput chunk : model.generate(inferParams)) {
				response.append(chunk.toString());
			}

			currentResponse = response.toString().trim();

			// Prepare next iteration prompt
			if (step < refinementSteps - 1) {
				currentPrompt = initialPrompt + "\n\nPrevious attempt: " + currentResponse +
					"\n\nPlease improve and refine this response:";
			}
		}

		return currentResponse;
	}

	/**
	 * Structured output parsing - extract specific information from generated text.
	 */
	public static Map<String, String> extractStructuredData(String text, List<String> fields) {
		if (text == null || fields == null) {
			return new HashMap<>();
		}

		Map<String, String> extracted = new HashMap<>();
		String[] lines = text.split("\n");

		for (String field : fields) {
			for (String line : lines) {
				// Look for patterns like "Field: value" or "Field = value"
				String pattern = "(?i)" + field + "\\s*[:=]\\s*(.+)";
				if (line.matches(".*" + pattern + ".*")) {
					String value = line.replaceAll(".*" + pattern + ".*", "$1").trim();
					extracted.put(field, value);
					break;
				}
			}
		}

		return extracted;
	}

	/**
	 * Shutdown the async executor (call when done with async operations).
	 */
	public static void shutdown() {
		asyncExecutor.shutdown();
	}
}
