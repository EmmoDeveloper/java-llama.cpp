package de.kherud.llama;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Ensemble inference engine that combines results from multiple models
 * using various voting and consensus strategies for improved accuracy
 * and reliability in AI IDE applications.
 */
public class EnsembleInferenceEngine {

	/**
	 * Voting strategies for ensemble decisions
	 */
	public enum VotingStrategy {
		MAJORITY_VOTE,        // Simple majority wins
		WEIGHTED_VOTE,        // Weight votes by model confidence/performance
		RANKED_CHOICE,        // Rank results and pick highest ranked
		CONFIDENCE_BASED,     // Weight by model-reported confidence
		CONSENSUS_THRESHOLD,  // Require minimum agreement threshold
		BEST_OF_N            // Pick best result from N models
	}

	/**
	 * Result aggregation strategies
	 */
	public enum AggregationStrategy {
		FIRST_SUCCESSFUL,     // Return first successful result
		BEST_QUALITY,         // Return highest quality result
		LONGEST_RESPONSE,     // Return longest response
		SHORTEST_RESPONSE,    // Return most concise response
		AVERAGED_SCORES,      // Average numerical scores/probabilities
		CONCATENATED         // Combine multiple results
	}

	/**
	 * Weighted result from a single model
	 */
	public static class WeightedResult<T> {
		public final T result;
		public final double weight;
		public final double confidence;
		public final String modelId;
		public final long responseTimeMs;
		public final Map<String, Object> metadata;

		public WeightedResult(T result, double weight, double confidence,
							  String modelId, long responseTimeMs) {
			this.result = result;
			this.weight = weight;
			this.confidence = confidence;
			this.modelId = modelId;
			this.responseTimeMs = responseTimeMs;
			this.metadata = new HashMap<>();
		}

		public WeightedResult<T> addMetadata(String key, Object value) {
			metadata.put(key, value);
			return this;
		}

		public double getScore() {
			return weight * confidence;
		}
	}

	/**
	 * Ensemble inference result with voting details
	 */
	public static class EnsembleResult<T> {
		public final T finalResult;
		public final List<WeightedResult<T>> allResults;
		public final VotingStrategy strategy;
		public final double consensusScore;
		public final Map<String, Object> votingMetadata;

		public EnsembleResult(T finalResult, List<WeightedResult<T>> allResults,
							  VotingStrategy strategy, double consensusScore) {
			this.finalResult = finalResult;
			this.allResults = new ArrayList<>(allResults);
			this.strategy = strategy;
			this.consensusScore = consensusScore;
			this.votingMetadata = new HashMap<>();
		}

		public EnsembleResult<T> addVotingMetadata(String key, Object value) {
			votingMetadata.put(key, value);
			return this;
		}

		public boolean hasConsensus(double threshold) {
			return consensusScore >= threshold;
		}

		public List<WeightedResult<T>> getTopResults(int n) {
			return allResults.stream()
				.sorted(Comparator.comparing((WeightedResult<T> r) -> r.getScore()).reversed())
				.limit(n)
				.collect(Collectors.toList());
		}
	}

	/**
	 * Code completion specific ensemble strategies
	 */
	public static class CodeCompletionEnsemble {

		/**
		 * Ensemble code completion with syntax validation
		 */
		public static EnsembleResult<String> completeCode(
				MultiModelManager modelManager,
				String codeContext,
				String language,
				VotingStrategy strategy) {

			MultiModelManager.RequestContext context = new MultiModelManager.RequestContext.Builder()
				.taskType("completion")
				.language(language)
				.requireCapability("code_generation")
				.build();

			// Execute completion on multiple models
			List<CompletableFuture<WeightedResult<String>>> futures = new ArrayList<>();

			// Get specialized code completion models
			List<String> codeModels = modelManager.getModelsBySpecialization(
				MultiModelManager.ModelSpecialization.CODE_COMPLETION);

			// If no models available, create mock results for demonstration
			if (codeModels.isEmpty()) {
				codeModels = List.of("mock-code-model-1", "mock-code-model-2");
			}

			for (String modelId : codeModels.subList(0, Math.min(3, codeModels.size()))) {
				CompletableFuture<WeightedResult<String>> future = CompletableFuture.supplyAsync(() -> {
					long startTime = System.currentTimeMillis();

					// Execute completion (simplified - would use actual model inference)
					String completion = executeCodeCompletion(modelId, codeContext);
					long responseTime = System.currentTimeMillis() - startTime;

					// Calculate weight based on model specialization and performance
					double weight = calculateCodeCompletionWeight(modelId, completion, language);
					double confidence = calculateCodeCompletionConfidence(completion, codeContext, language);

					return new WeightedResult<>(completion, weight, confidence, modelId, responseTime)
						.addMetadata("syntaxValid", isValidSyntax(completion, language))
						.addMetadata("completionLength", completion.length());
				});

				futures.add(future);
			}

			// Wait for all completions
			List<WeightedResult<String>> results = futures.stream()
				.map(CompletableFuture::join)
				.collect(Collectors.toList());

			// Apply ensemble voting
			return applyVotingStrategy(results, strategy);
		}

		private static String executeCodeCompletion(String modelId, String codeContext) {
			// Simplified completion - in real implementation, this would use the actual model
			return "// Generated by " + modelId + "\nfunction completeThis() {\n    return true;\n}";
		}

		private static double calculateCodeCompletionWeight(String modelId, String completion, String language) {
			double weight = 1.0;

			// Boost weight for language-specific models
			if (modelId.toLowerCase().contains(language.toLowerCase())) {
				weight += 0.3;
			}

			// Boost weight for completion quality indicators
			if (completion.contains("function") || completion.contains("class")) {
				weight += 0.2;
			}

			// Penalize very short or very long completions
			if (completion.length() < 10 || completion.length() > 1000) {
				weight -= 0.2;
			}

			return Math.max(0.1, weight);
		}

		private static double calculateCodeCompletionConfidence(String completion, String context, String language) {
			double confidence = 0.5; // Base confidence

			// Increase confidence for syntactically valid code
			if (isValidSyntax(completion, language)) {
				confidence += 0.3;
			}

			// Increase confidence for contextually relevant completion
			if (isContextuallyRelevant(completion, context)) {
				confidence += 0.2;
			}

			return Math.min(1.0, confidence);
		}

		private static boolean isValidSyntax(String code, String language) {
			// Simplified syntax validation - would use actual parsers
			switch (language.toLowerCase()) {
				case "java":
					return code.contains("{") && code.contains("}") && !code.contains("syntax error");
				case "python":
					return !code.contains("syntax error") && code.trim().length() > 0;
				case "javascript":
					return code.contains("{") || code.contains("function") || code.contains("=>");
				default:
					return code.trim().length() > 0;
			}
		}

		private static boolean isContextuallyRelevant(String completion, String context) {
			// Simple relevance check - would use more sophisticated analysis
			String[] contextWords = context.toLowerCase().split("\\W+");
			String completionLower = completion.toLowerCase();

			return Arrays.stream(contextWords)
				.filter(word -> word.length() > 3)
				.anyMatch(completionLower::contains);
		}
	}

	/**
	 * JSON generation ensemble with schema validation
	 */
	public static class JsonGenerationEnsemble {

		public static EnsembleResult<String> generateJson(
				MultiModelManager modelManager,
				String schema,
				String prompt,
				VotingStrategy strategy) {

			MultiModelManager.RequestContext context = new MultiModelManager.RequestContext.Builder()
				.taskType("json_generation")
				.requireCapability("structured_output")
				.build();

			// Execute on multiple models
			CompletableFuture<List<String>> futureResults =
				modelManager.executeEnsemble(context, model -> {
					// Generate JSON using constrained sampling
					try (JsonConstrainedSampler jsonSampler = new JsonConstrainedSampler()) {
						return generateJsonWithModel(model, schema, prompt, jsonSampler);
					}
				}, 3);

			List<WeightedResult<String>> results = futureResults.join().stream()
				.map(json -> new WeightedResult<>(json, 1.0, calculateJsonConfidence(json, schema),
												  "model", 0L))
				.collect(Collectors.toList());

			return applyVotingStrategy(results, strategy);
		}

		private static String generateJsonWithModel(LlamaModel model, String schema,
													String prompt, JsonConstrainedSampler sampler) {
			// Simplified JSON generation - would use actual model inference with constraints
			return "{\n  \"generated\": true,\n  \"schema\": \"" + schema + "\",\n  \"prompt\": \"" +
				   prompt + "\"\n}";
		}

		private static double calculateJsonConfidence(String json, String schema) {
			double confidence = 0.3; // Base confidence

			try {
				// Check if valid JSON
				new com.fasterxml.jackson.databind.ObjectMapper().readTree(json);
				confidence += 0.4;

				// Check if matches schema (simplified)
				if (schema != null && json.contains("\"")) {
					confidence += 0.3;
				}
			} catch (Exception e) {
				// Invalid JSON - low confidence
				confidence = 0.1;
			}

			return confidence;
		}
	}

	/**
	 * Apply voting strategy to weighted results
	 */
	public static <T> EnsembleResult<T> applyVotingStrategy(
			List<WeightedResult<T>> results, VotingStrategy strategy) {

		if (results.isEmpty()) {
			throw new IllegalArgumentException("No results to vote on");
		}

		switch (strategy) {
			case MAJORITY_VOTE:
				return majorityVote(results, strategy);

			case WEIGHTED_VOTE:
				return weightedVote(results, strategy);

			case CONFIDENCE_BASED:
				return confidenceBasedVote(results, strategy);

			case BEST_OF_N:
				return bestOfN(results, strategy);

			case CONSENSUS_THRESHOLD:
				return consensusThreshold(results, strategy, 0.6);

			case RANKED_CHOICE:
			default:
				return rankedChoice(results, strategy);
		}
	}

	private static <T> EnsembleResult<T> majorityVote(List<WeightedResult<T>> results, VotingStrategy strategy) {
		// Group by result content and count occurrences
		Map<T, Long> voteCounts = results.stream()
			.collect(Collectors.groupingBy(r -> r.result, Collectors.counting()));

		T winner = voteCounts.entrySet().stream()
			.max(Map.Entry.comparingByValue())
			.map(Map.Entry::getKey)
			.orElse(results.get(0).result);

		double consensusScore = (double) voteCounts.get(winner) / results.size();

		return new EnsembleResult<>(winner, results, strategy, consensusScore)
			.addVotingMetadata("voteCount", voteCounts.get(winner))
			.addVotingMetadata("totalVotes", results.size());
	}

	private static <T> EnsembleResult<T> weightedVote(List<WeightedResult<T>> results, VotingStrategy strategy) {
		// Group by result and sum weights
		Map<T, Double> weightSums = new HashMap<>();
		for (WeightedResult<T> result : results) {
			weightSums.merge(result.result, result.weight, Double::sum);
		}

		T winner = weightSums.entrySet().stream()
			.max(Map.Entry.comparingByValue())
			.map(Map.Entry::getKey)
			.orElse(results.get(0).result);

		double totalWeight = weightSums.values().stream().mapToDouble(Double::doubleValue).sum();
		double consensusScore = weightSums.get(winner) / totalWeight;

		return new EnsembleResult<>(winner, results, strategy, consensusScore)
			.addVotingMetadata("winnerWeight", weightSums.get(winner))
			.addVotingMetadata("totalWeight", totalWeight);
	}

	private static <T> EnsembleResult<T> confidenceBasedVote(List<WeightedResult<T>> results, VotingStrategy strategy) {
		WeightedResult<T> mostConfident = results.stream()
			.max(Comparator.comparing(r -> r.confidence))
			.orElse(results.get(0));

		double avgConfidence = results.stream()
			.mapToDouble(r -> r.confidence)
			.average()
			.orElse(0.0);

		return new EnsembleResult<>(mostConfident.result, results, strategy, mostConfident.confidence)
			.addVotingMetadata("averageConfidence", avgConfidence)
			.addVotingMetadata("maxConfidence", mostConfident.confidence);
	}

	private static <T> EnsembleResult<T> bestOfN(List<WeightedResult<T>> results, VotingStrategy strategy) {
		WeightedResult<T> best = results.stream()
			.max(Comparator.comparing(WeightedResult::getScore))
			.orElse(results.get(0));

		return new EnsembleResult<>(best.result, results, strategy, best.getScore())
			.addVotingMetadata("bestScore", best.getScore())
			.addVotingMetadata("bestModelId", best.modelId);
	}

	private static <T> EnsembleResult<T> consensusThreshold(
			List<WeightedResult<T>> results, VotingStrategy strategy, double threshold) {

		EnsembleResult<T> majorityResult = majorityVote(results, strategy);

		if (majorityResult.consensusScore >= threshold) {
			return majorityResult;
		} else {
			// Fall back to best scoring result if no consensus
			return bestOfN(results, strategy)
				.addVotingMetadata("consensusReached", false)
				.addVotingMetadata("consensusThreshold", threshold);
		}
	}

	private static <T> EnsembleResult<T> rankedChoice(List<WeightedResult<T>> results, VotingStrategy strategy) {
		// Rank by combined score (weight * confidence)
		List<WeightedResult<T>> ranked = results.stream()
			.sorted(Comparator.comparing((WeightedResult<T> r) -> r.getScore()).reversed())
			.collect(Collectors.toList());

		WeightedResult<T> winner = ranked.get(0);
		double consensusScore = winner.getScore();

		return new EnsembleResult<>(winner.result, results, strategy, consensusScore)
			.addVotingMetadata("ranking", ranked.stream()
				.map(r -> r.modelId + ":" + String.format("%.3f", r.getScore()))
				.collect(Collectors.toList()));
	}
}