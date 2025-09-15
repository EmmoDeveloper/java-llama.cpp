package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;
import java.util.*;

public class BatchOptimizationTest {
	private static final System.Logger logger = System.getLogger(BatchOptimizationTest.class.getName());

	@Test
	public void testBatchSizeOptimization() {
		logger.log(DEBUG, "\n=== Batch Size Optimization Investigation ===\n");

		// Test different batch sizes to find optimal throughput
		int[] batchSizes = {128, 256, 512, 1024, 2048};
		String testPrompt = "def binary_search(arr, target):";
		int nPredict = 20;

		List<BatchResult> results = new ArrayList<>();

		for (int batchSize : batchSizes) {
			logger.log(DEBUG, "\n🔍 Testing batch size: %d", batchSize);
			BatchResult result = benchmarkBatchSize(batchSize, testPrompt, nPredict);
			results.add(result);

			logger.log(DEBUG, "   Throughput: %.2f tokens/second", result.tokensPerSecond);
			logger.log(DEBUG, "   Latency: %d ms", result.latencyMs);
			logger.log(DEBUG, "   Memory efficiency: %s", result.memoryEfficient ? "Good" : "Poor");
		}

		// Analyze results
		logger.log(DEBUG, "\n=== Batch Size Analysis ===");
		BatchResult best = findOptimalBatchSize(results);

		logger.log(DEBUG, "🏆 Optimal batch size: %d", best.batchSize);
		logger.log(DEBUG, "   Best throughput: %.2f tokens/second", best.tokensPerSecond);
		logger.log(DEBUG, "   Latency: %d ms", best.latencyMs);

		// Compare with current default (512)
		BatchResult defaultResult = results.stream()
			.filter(r -> r.batchSize == 512)
			.findFirst()
			.orElse(null);

		if (defaultResult != null && best.batchSize != 512) {
			double improvement = (best.tokensPerSecond - defaultResult.tokensPerSecond) / defaultResult.tokensPerSecond * 100;
			logger.log(DEBUG, "📈 Improvement over current default (512): %.1f%%", improvement);
		}

		// Test ubatch size variations
		logger.log(DEBUG, "\n=== UBatch Size Investigation ===");
		testUbatchSizeVariations(best.batchSize);

		// Verify results are reasonable
		Assert.assertTrue("Should test multiple batch sizes", results.size() >= 3);
		Assert.assertTrue("Best batch size should show good performance",
			best.tokensPerSecond > 5.0); // At least 5 tokens/second
		Assert.assertTrue("All results should have positive throughput",
			results.stream().allMatch(r -> r.tokensPerSecond > 0));

		logger.log(DEBUG, "\n✅ Batch optimization investigation completed!");
	}

	@Test
	public void testContextSizeImpactOnBatching() {
		logger.log(DEBUG, "\n=== Context Size Impact on Batch Performance ===\n");

		// Test how different context sizes affect optimal batch size
		int[] contextSizes = {512, 1024, 2048, 4096};
		int fixedBatchSize = 512;
		String testPrompt = "class QuickSort:";
		int nPredict = 15;

		for (int ctxSize : contextSizes) {
			logger.log(DEBUG, "\n📏 Testing context size: %d", ctxSize);

			try (LlamaModel model = new LlamaModel(
				new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setCtxSize(ctxSize)
					.setBatchSize(fixedBatchSize)
					.setGpuLayers(43)
			)) {

				InferenceParameters params = new InferenceParameters(testPrompt)
					.setNPredict(nPredict)
					.setTemperature(0.1f);

				long startTime = System.currentTimeMillis();
				int tokenCount = 0;

				for (LlamaOutput output : model.generate(params)) {
					tokenCount++;
				}

				long duration = System.currentTimeMillis() - startTime;
				double tokensPerSecond = tokenCount * 1000.0 / duration;

				logger.log(DEBUG, "   Throughput: %.2f tokens/second", tokensPerSecond);
				logger.log(DEBUG, "   Latency: %d ms", duration);

				// Context size affects memory usage and potentially batch efficiency
				Assert.assertTrue("Should generate tokens", tokenCount > 0);

			}
		}

		logger.log(DEBUG, "\n✅ Context size impact analysis completed!");
	}

	@Test
	public void testOptimalBatchConfiguration() {
		logger.log(DEBUG, "\n=== Testing Optimized Batch Configuration ===\n");

		// Test the current smart defaults against potentially better configurations
		String testPrompt = "import numpy as np\ndef matrix_multiply(A, B):";
		int nPredict = 25;

		logger.log(DEBUG, "1. Current Smart Defaults:");
		long defaultTime = benchmarkConfiguration("Current", null, testPrompt, nPredict);

		logger.log(DEBUG, "\n2. Optimized Configuration (Large Batch):");
		ModelParameters optimized1 = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setBatchSize(1024)
			.setUbatchSize(512)
			.setCtxSize(2048)
			.setGpuLayers(43);
		long optimized1Time = benchmarkConfiguration("Large Batch", optimized1, testPrompt, nPredict);

		logger.log(DEBUG, "\n3. Optimized Configuration (Medium Batch):");
		ModelParameters optimized2 = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setBatchSize(256)
			.setUbatchSize(256)
			.setCtxSize(2048)
			.setGpuLayers(43);
		long optimized2Time = benchmarkConfiguration("Medium Batch", optimized2, testPrompt, nPredict);

		// Analysis
		logger.log(DEBUG, "\n=== Configuration Comparison ===");
		logger.log(DEBUG, "Current defaults: %d ms", defaultTime);
		logger.log(DEBUG, "Large batch config: %d ms", optimized1Time);
		logger.log(DEBUG, "Medium batch config: %d ms", optimized2Time);

		long bestTime = Math.min(Math.min(defaultTime, optimized1Time), optimized2Time);

		if (bestTime == defaultTime) {
			logger.log(DEBUG, "🎯 Current defaults are already optimal!");
		} else if (bestTime == optimized1Time) {
			logger.log(DEBUG, "📈 Large batch configuration is better!");
		} else {
			logger.log(DEBUG, "⚡ Medium batch configuration is better!");
		}

		logger.log(DEBUG, "\n✅ Batch configuration optimization completed!");
	}

	private BatchResult benchmarkBatchSize(int batchSize, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setBatchSize(batchSize)
				.setCtxSize(2048)
				.setGpuLayers(43)
		)) {

			InferenceParameters params = new InferenceParameters(prompt)
				.setNPredict(nPredict)
				.setTemperature(0.1f);

			long startTime = System.currentTimeMillis();
			int tokenCount = 0;

			for (LlamaOutput output : model.generate(params)) {
				tokenCount++;
			}

			long duration = System.currentTimeMillis() - startTime;
			double tokensPerSecond = tokenCount * 1000.0 / duration;

			// Simple heuristic for memory efficiency (larger batches may be less efficient)
			boolean memoryEfficient = batchSize <= 1024;

			return new BatchResult(batchSize, tokensPerSecond, duration, memoryEfficient);

		}
	}

	private void testUbatchSizeVariations(int optimalBatchSize) {
		int[] ubatchSizes = {128, 256, 512, optimalBatchSize};
		String testPrompt = "function fibonacci(n) {";
		int nPredict = 10;

		for (int ubatchSize : ubatchSizes) {
			if (ubatchSize > optimalBatchSize) continue; // ubatch should be <= batch

			logger.log(DEBUG, "  Testing ubatch size: %d (batch: %d)", ubatchSize, optimalBatchSize);

			try (LlamaModel model = new LlamaModel(
				new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setBatchSize(optimalBatchSize)
					.setUbatchSize(ubatchSize)
					.setCtxSize(1024)
					.setGpuLayers(43)
			)) {

				InferenceParameters params = new InferenceParameters(testPrompt)
					.setNPredict(nPredict)
					.setTemperature(0.1f);

				long startTime = System.currentTimeMillis();
				int tokenCount = 0;

				for (LlamaOutput output : model.generate(params)) {
					tokenCount++;
				}

				long duration = System.currentTimeMillis() - startTime;
				double tokensPerSecond = tokenCount * 1000.0 / duration;

				logger.log(DEBUG, "    Throughput: %.2f tokens/second", tokensPerSecond);

			}
		}
	}

	private long benchmarkConfiguration(String name, ModelParameters params, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(Objects.requireNonNullElseGet(params, () -> new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")))) {
			// Use smart defaults

			InferenceParameters inferenceParams = new InferenceParameters(prompt)
				.setNPredict(nPredict)
				.setTemperature(0.1f);

			long startTime = System.currentTimeMillis();
			int tokenCount = 0;

			for (LlamaOutput output : model.generate(inferenceParams)) {
				tokenCount++;
			}

			long duration = System.currentTimeMillis() - startTime;
			double tokensPerSecond = tokenCount * 1000.0 / duration;

			logger.log(DEBUG, "   %s: %d tokens in %d ms (%.2f tok/s)",
				name, tokenCount, duration, tokensPerSecond);

			return duration;

		}
	}

	private BatchResult findOptimalBatchSize(List<BatchResult> results) {
		// Find batch size with best throughput while considering memory efficiency
		return results.stream()
			.max(Comparator.comparing(r -> r.tokensPerSecond * (r.memoryEfficient ? 1.1 : 1.0)))
			.orElse(results.get(0));
	}

	private record BatchResult(int batchSize, double tokensPerSecond, long latencyMs, boolean memoryEfficient) {
	}
}
