package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;
import java.util.*;

public class BatchOptimizationTest {

	@Test
	public void testBatchSizeOptimization() {
		System.out.println("\n=== Batch Size Optimization Investigation ===\n");

		// Test different batch sizes to find optimal throughput
		int[] batchSizes = {128, 256, 512, 1024, 2048};
		String testPrompt = "def binary_search(arr, target):";
		int nPredict = 20;

		List<BatchResult> results = new ArrayList<>();

		for (int batchSize : batchSizes) {
			System.out.printf("\n🔍 Testing batch size: %d\n", batchSize);
			BatchResult result = benchmarkBatchSize(batchSize, testPrompt, nPredict);
			results.add(result);

			System.out.printf("   Throughput: %.2f tokens/second\n", result.tokensPerSecond);
			System.out.printf("   Latency: %d ms\n", result.latencyMs);
			System.out.printf("   Memory efficiency: %s\n", result.memoryEfficient ? "Good" : "Poor");
		}

		// Analyze results
		System.out.println("\n=== Batch Size Analysis ===");
		BatchResult best = findOptimalBatchSize(results);

		System.out.printf("🏆 Optimal batch size: %d\n", best.batchSize);
		System.out.printf("   Best throughput: %.2f tokens/second\n", best.tokensPerSecond);
		System.out.printf("   Latency: %d ms\n", best.latencyMs);

		// Compare with current default (512)
		BatchResult defaultResult = results.stream()
			.filter(r -> r.batchSize == 512)
			.findFirst()
			.orElse(null);

		if (defaultResult != null && best.batchSize != 512) {
			double improvement = (best.tokensPerSecond - defaultResult.tokensPerSecond) / defaultResult.tokensPerSecond * 100;
			System.out.printf("📈 Improvement over current default (512): %.1f%%\n", improvement);
		}

		// Test ubatch size variations
		System.out.println("\n=== UBatch Size Investigation ===");
		testUbatchSizeVariations(best.batchSize);

		// Verify results are reasonable
		Assert.assertTrue("Should test multiple batch sizes", results.size() >= 3);
		Assert.assertTrue("Best batch size should show good performance",
			best.tokensPerSecond > 5.0); // At least 5 tokens/second
		Assert.assertTrue("All results should have positive throughput",
			results.stream().allMatch(r -> r.tokensPerSecond > 0));

		System.out.println("\n✅ Batch optimization investigation completed!");
	}

	@Test
	public void testContextSizeImpactOnBatching() {
		System.out.println("\n=== Context Size Impact on Batch Performance ===\n");

		// Test how different context sizes affect optimal batch size
		int[] contextSizes = {512, 1024, 2048, 4096};
		int fixedBatchSize = 512;
		String testPrompt = "class QuickSort:";
		int nPredict = 15;

		for (int ctxSize : contextSizes) {
			System.out.printf("\n📏 Testing context size: %d\n", ctxSize);

			try (LlamaModel model = new LlamaModel(
				new ModelParameters()
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
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

				System.out.printf("   Throughput: %.2f tokens/second\n", tokensPerSecond);
				System.out.printf("   Latency: %d ms\n", duration);

				// Context size affects memory usage and potentially batch efficiency
				Assert.assertTrue("Should generate tokens", tokenCount > 0);

			}
		}

		System.out.println("\n✅ Context size impact analysis completed!");
	}

	@Test
	public void testOptimalBatchConfiguration() {
		System.out.println("\n=== Testing Optimized Batch Configuration ===\n");

		// Test the current smart defaults against potentially better configurations
		String testPrompt = "import numpy as np\ndef matrix_multiply(A, B):";
		int nPredict = 25;

		System.out.println("1. Current Smart Defaults:");
		long defaultTime = benchmarkConfiguration("Current", null, testPrompt, nPredict);

		System.out.println("\n2. Optimized Configuration (Large Batch):");
		ModelParameters optimized1 = new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			.setBatchSize(1024)
			.setUbatchSize(512)
			.setCtxSize(2048)
			.setGpuLayers(43);
		long optimized1Time = benchmarkConfiguration("Large Batch", optimized1, testPrompt, nPredict);

		System.out.println("\n3. Optimized Configuration (Medium Batch):");
		ModelParameters optimized2 = new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			.setBatchSize(256)
			.setUbatchSize(256)
			.setCtxSize(2048)
			.setGpuLayers(43);
		long optimized2Time = benchmarkConfiguration("Medium Batch", optimized2, testPrompt, nPredict);

		// Analysis
		System.out.println("\n=== Configuration Comparison ===");
		System.out.printf("Current defaults: %d ms\n", defaultTime);
		System.out.printf("Large batch config: %d ms\n", optimized1Time);
		System.out.printf("Medium batch config: %d ms\n", optimized2Time);

		long bestTime = Math.min(Math.min(defaultTime, optimized1Time), optimized2Time);

		if (bestTime == defaultTime) {
			System.out.println("🎯 Current defaults are already optimal!");
		} else if (bestTime == optimized1Time) {
			System.out.println("📈 Large batch configuration is better!");
		} else {
			System.out.println("⚡ Medium batch configuration is better!");
		}

		System.out.println("\n✅ Batch configuration optimization completed!");
	}

	private BatchResult benchmarkBatchSize(int batchSize, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
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

			System.out.printf("  Testing ubatch size: %d (batch: %d)\n", ubatchSize, optimalBatchSize);

			try (LlamaModel model = new LlamaModel(
				new ModelParameters()
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
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

				System.out.printf("    Throughput: %.2f tokens/second\n", tokensPerSecond);

			}
		}
	}

	private long benchmarkConfiguration(String name, ModelParameters params, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(Objects.requireNonNullElseGet(params, () -> new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")))) {
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

			System.out.printf("   %s: %d tokens in %d ms (%.2f tok/s)\n",
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
