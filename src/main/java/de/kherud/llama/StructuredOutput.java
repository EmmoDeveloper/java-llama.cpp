package de.kherud.llama;

import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

/**
 * Service for generating structured outputs using a LlamaModel.
 * Uses static utilities from StructuredOutputUtils.
 */
public class StructuredOutput {

	private final LlamaModel model;

	public StructuredOutput(LlamaModel model) {
		this.model = model;
	}

	/**
	 * Generate JSON output with schema validation
	 */
	public StructuredOutputUtils.JsonResult generateJson(String prompt, StructuredOutputUtils.JsonSchema schema) {
		return generateJson(prompt, schema, new InferenceParameters(""));
	}

	/**
	 * Generate JSON output with custom parameters
	 */
	public StructuredOutputUtils.JsonResult generateJson(String prompt, StructuredOutputUtils.JsonSchema schema, InferenceParameters params) {
		// Convert schema to GBNF grammar using static method
		String grammar = LlamaModel.jsonSchemaToGrammar(schema.toJsonString());
		params.setGrammar(grammar);

		// Add JSON instruction to prompt
		String jsonPrompt = StructuredOutputUtils.formatJsonPrompt(prompt, schema);
		params.setPrompt(jsonPrompt);

		// Generate response
		String response = model.complete(params);

		// Extract and validate JSON
		return StructuredOutputUtils.parseJsonResponse(response, schema);
	}

	/**
	 * Async version of generateJson
	 */
	public CompletableFuture<StructuredOutputUtils.JsonResult> generateJsonAsync(String prompt, StructuredOutputUtils.JsonSchema schema) {
		return CompletableFuture.supplyAsync(() -> generateJson(prompt, schema));
	}

	/**
	 * Generate function call
	 */
	public StructuredOutputUtils.FunctionCallResult generateFunctionCall(String prompt, List<StructuredOutputUtils.FunctionSpec> functions) {
		return generateFunctionCall(prompt, functions, new InferenceParameters(""));
	}

	/**
	 * Generate function call with custom parameters
	 */
	public StructuredOutputUtils.FunctionCallResult generateFunctionCall(String prompt, List<StructuredOutputUtils.FunctionSpec> functions,
																		InferenceParameters params) {
		// Format prompt with function specifications
		String functionPrompt = StructuredOutputUtils.formatFunctionPrompt(prompt, functions);
		params.setPrompt(functionPrompt);

		// Generate response
		String response = model.complete(params);

		// Parse function call
		return StructuredOutputUtils.parseFunctionCall(response, functions);
	}

	/**
	 * Generate multiple outputs and select best
	 */
	public String generateBestOf(String prompt, int n, OutputScorer scorer) {
		List<String> candidates = new ArrayList<>();

		for (int i = 0; i < n; i++) {
			InferenceParameters params = new InferenceParameters(prompt);
			params.setTemperature(0.7f + (i * 0.1f)); // Vary temperature
			candidates.add(model.complete(params));
		}

		return scorer.selectBest(candidates);
	}

	/**
	 * Interface for scoring outputs
	 */
	public interface OutputScorer {
		String selectBest(List<String> candidates);
	}

	/**
	 * Simple length-based scorer
	 */
	public static class LengthScorer implements OutputScorer {
		private final boolean preferLonger;

		public LengthScorer(boolean preferLonger) {
			this.preferLonger = preferLonger;
		}

		@Override
		public String selectBest(List<String> candidates) {
			return candidates.stream()
				.max((a, b) -> preferLonger ?
					Integer.compare(a.length(), b.length()) :
					Integer.compare(b.length(), a.length()))
				.orElse("");
		}
	}
}