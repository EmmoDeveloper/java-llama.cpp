package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

public class QuickBatchTest {

	@Test
	public void testOptimalBatchSizes() {
		System.out.println("\n=== Quick Batch Size Investigation ===\n");

		// Test fewer batch sizes with shorter prompts for faster results
		int[] batchSizes = {128, 512, 1024};
		String prompt = "def test():";
		int nPredict = 8; // Short generation for speed

		System.out.println("🔍 Comparing batch sizes for throughput optimization:\n");

		BatchResult best = null;
		double bestThroughput = 0;

		for (int batchSize : batchSizes) {
			System.out.printf("Testing batch size %d... ", batchSize);

			BatchResult result = quickBenchmark(batchSize, prompt, nPredict);
			System.out.printf("%.1f tok/s (%.0f ms)\n",
				result.throughput, result.latency);

			if (result.throughput > bestThroughput) {
				bestThroughput = result.throughput;
				best = result;
			}
		}

		Assert.assertNotNull(best);
		System.out.printf("\n🏆 Optimal batch size: %d (%.1f tokens/second)\n",
			best.batchSize, best.throughput);

		// Test ubatch size impact on the optimal batch size
		System.out.printf("\n🔬 Testing ubatch variations for batch size %d:\n", best.batchSize);
		testUbatchVariations(best.batchSize, prompt, nPredict);

		System.out.println("\n✅ Quick batch optimization completed!");
	}

	@Test
	public void testCurrentDefaultPerformance() {
		System.out.println("\n=== Current Default vs Optimized Batch Configuration ===\n");

		String prompt = "import json";
		int nPredict = 10;

		// Current default (smart defaults)
		System.out.print("Current smart defaults: ");
		BatchResult currentDefault = benchmarkWithDefaults(prompt, nPredict);
		System.out.printf("%.1f tok/s\n", currentDefault.throughput);

		// Test potentially better configuration
		System.out.print("Large batch (1024): ");
		BatchResult largeBatch = benchmarkWithConfig(1024, 512, prompt, nPredict);
		System.out.printf("%.1f tok/s\n", largeBatch.throughput);

		System.out.print("Small batch (256): ");
		BatchResult smallBatch = benchmarkWithConfig(256, 256, prompt, nPredict);
		System.out.printf("%.1f tok/s\n", smallBatch.throughput);

		// Analysis
		double improvement1 = (largeBatch.throughput - currentDefault.throughput) / currentDefault.throughput * 100;
		double improvement2 = (smallBatch.throughput - currentDefault.throughput) / currentDefault.throughput * 100;

		System.out.print("\n📊 Analysis:\n");
		System.out.printf("   Large batch improvement: %+.1f%%\n", improvement1);
		System.out.printf("   Small batch improvement: %+.1f%%\n", improvement2);

		if (Math.abs(improvement1) < 5 && Math.abs(improvement2) < 5) {
			System.out.println("✅ Current defaults are well-optimized (within 5% of alternatives)");
		} else if (improvement1 > improvement2 && improvement1 > 5) {
			System.out.println("📈 Large batch configuration shows significant improvement!");
		} else if (improvement2 > 5) {
			System.out.println("📈 Small batch configuration shows significant improvement!");
		}

		System.out.println("\n✅ Batch configuration analysis completed!");
	}

	private BatchResult quickBenchmark(int batchSize, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setBatchSize(batchSize)
				.setCtxSize(1024) // Smaller context for speed
				.setGpuLayers(43)
		)) {
			// Smaller context for speed

			return executeTest(model, prompt, nPredict, batchSize);

		}
	}

	private BatchResult benchmarkWithDefaults(String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setCtxSize(1024)
		)) {
			// Use smart defaults

			return executeTest(model, prompt, nPredict, 512); // Assume 512 default

		}
	}

	private BatchResult benchmarkWithConfig(int batchSize, int ubatchSize, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setBatchSize(batchSize)
				.setUbatchSize(ubatchSize)
				.setCtxSize(1024)
				.setGpuLayers(43)
		)) {

			return executeTest(model, prompt, nPredict, batchSize);

		}
	}

	private void testUbatchVariations(int batchSize, String prompt, int nPredict) {
		int[] ubatchSizes = {128, 256, 512};

		for (int ubatchSize : ubatchSizes) {
			if (ubatchSize > batchSize) continue;

			System.out.printf("   ubatch=%d: ", ubatchSize);
			BatchResult result = benchmarkWithConfig(batchSize, ubatchSize, prompt, nPredict);
			System.out.printf("%.1f tok/s\n", result.throughput);
		}
	}

	private BatchResult executeTest(LlamaModel model, String prompt, int nPredict, int batchSize) {
		InferenceParameters params = new InferenceParameters(prompt)
			.setNPredict(nPredict)
			.setTemperature(0.1f);

		long startTime = System.currentTimeMillis();
		int tokenCount = 0;

		for (LlamaOutput output : model.generate(params)) {
			tokenCount++;
		}

		long duration = System.currentTimeMillis() - startTime;
		double throughput = tokenCount * 1000.0 / duration;

		return new BatchResult(batchSize, throughput, duration);
	}

	private record BatchResult(int batchSize, double throughput, double latency) {
	}
}
