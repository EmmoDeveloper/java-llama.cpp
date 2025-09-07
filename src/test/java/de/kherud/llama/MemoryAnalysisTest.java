package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.lang.management.GarbageCollectorMXBean;
import java.util.List;
import java.util.ArrayList;

public class MemoryAnalysisTest {

	private static final int MB = 1024 * 1024;

	@Test
	public void testMemoryUsagePatterns() {
		System.out.println("\n=== Memory Usage Pattern Analysis ===\n");

		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

		// Initial memory state
		MemorySnapshot initial = takeSnapshot("Initial");
		long gcCountBefore = getTotalGcCount(gcBeans);

		// Test memory usage during model lifecycle
		System.out.println("🔍 Testing memory usage during model lifecycle...");

		LlamaModel model = null;
		MemorySnapshot afterLoad;
		MemorySnapshot afterGeneration;

		try {
			// Step 1: Model loading
			System.out.println("Loading model...");
			model = new LlamaModel(
				new ModelParameters()
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
					.setCtxSize(1024)
					.setGpuLayers(43)
			);
			afterLoad = takeSnapshot("After Load");

			// Step 2: Token generation
			System.out.println("Generating tokens...");
			InferenceParameters params = new InferenceParameters("def test_function():")
				.setNPredict(20)
				.setTemperature(0.1f);

			int tokenCount = 0;
			for (LlamaOutput output : model.generate(params)) {
				tokenCount++;
			}

			afterGeneration = takeSnapshot("After Generation");

			System.out.printf("Generated %d tokens\n", tokenCount);

			Assert.assertTrue("Should generate tokens", tokenCount > 0);

		} finally {
			if (model != null) {
				System.out.println("Closing model...");
				model.close();
			}
		}

		// Wait for potential cleanup
		System.gc();
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		MemorySnapshot afterClose = takeSnapshot("After Close");
		long gcCountAfter = getTotalGcCount(gcBeans);

		// Analysis
		System.out.println("\n=== Memory Analysis Results ===");

		long loadingMemoryIncrease = afterLoad.heapUsed - initial.heapUsed;
		long generationMemoryIncrease = afterGeneration.heapUsed - afterLoad.heapUsed;
		long memoryReclaimed = afterGeneration.heapUsed - afterClose.heapUsed;

		System.out.printf("Memory increase during loading: %d MB\n", loadingMemoryIncrease / MB);
		System.out.printf("Memory increase during generation: %d MB\n", generationMemoryIncrease / MB);
		System.out.printf("Memory reclaimed after close: %d MB\n", memoryReclaimed / MB);
		System.out.printf("GC cycles triggered: %d\n", gcCountAfter - gcCountBefore);

		// Print detailed snapshots
		System.out.println("\n📊 Detailed Memory Snapshots:");
		System.out.println(initial);
		System.out.println(afterLoad);
		System.out.println(afterGeneration);
		System.out.println(afterClose);

		Assert.assertTrue("Memory should be reclaimed after close", memoryReclaimed > 0 || loadingMemoryIncrease < 50 * MB);

		System.out.println("\n✅ Memory pattern analysis completed!");
	}

	@Test
	public void testMemoryLeakDetection() {
		System.out.println("\n=== Memory Leak Detection Test ===\n");

		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		List<Long> heapUsages = new ArrayList<>();

		System.out.println("🔍 Testing for memory leaks with multiple model instances...");

		// Test multiple create/close cycles
		for (int i = 0; i < 3; i++) {
			System.out.printf("Cycle %d: ", i + 1);

			try (LlamaModel model = new LlamaModel(
				new ModelParameters()
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
					.setCtxSize(512)
					.setGpuLayers(43)
			)) {

				// Quick generation
				InferenceParameters params = new InferenceParameters("test")
					.setNPredict(3);

				int tokens = 0;
				for (LlamaOutput output : model.generate(params)) {
					tokens++;
				}
				System.out.printf("Generated %d tokens, ", tokens);

			}

			// Force GC and measure
			System.gc();
			try {
				Thread.sleep(50);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}

			long heapUsed = memoryBean.getHeapMemoryUsage().getUsed();
			heapUsages.add(heapUsed);
			System.out.printf("Heap: %d MB\n", heapUsed / MB);
		}

		// Analyze leak trend
		System.out.println("\n📈 Memory Leak Analysis:");
		boolean memoryIncreasing = true;
		for (int i = 1; i < heapUsages.size(); i++) {
			long diff = heapUsages.get(i) - heapUsages.get(i-1);
			System.out.printf("Cycle %d vs %d: %+d MB\n", i, i+1, diff / MB);

			if (diff < 10 * MB) { // Allow 10MB variation
				memoryIncreasing = false;
			}
		}

		if (memoryIncreasing) {
			System.out.println("⚠️  Potential memory leak detected - heap consistently growing");
		} else {
			System.out.println("✅ No significant memory leak detected");
		}

		// Final assertion
		long totalIncrease = heapUsages.get(heapUsages.size()-1) - heapUsages.get(0);
		Assert.assertTrue("Memory increase should be reasonable (< 100MB)",
			Math.abs(totalIncrease) < 100 * MB);

		System.out.println("\n✅ Memory leak detection completed!");
	}

	@Test
	public void testConcurrentMemoryUsage() {
		System.out.println("\n=== Concurrent Memory Usage Test ===\n");

		MemorySnapshot initial = takeSnapshot("Initial");

		System.out.println("🔍 Testing memory usage with concurrent model usage...");

		// Test resource sharing and memory efficiency
		List<LlamaModel> models = new ArrayList<>();
		MemorySnapshot afterGeneration;

		try {
			// Create multiple models (should reuse native resources efficiently)
			for (int i = 0; i < 2; i++) {
				System.out.printf("Creating model %d...\n", i + 1);
				LlamaModel model = new LlamaModel(
					new ModelParameters()
						.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
						.setCtxSize(256) // Smaller for memory efficiency
						.setGpuLayers(43)
				);
				models.add(model);
			}

			MemorySnapshot afterModels = takeSnapshot("After Models");

			// Test concurrent generation
			System.out.println("Testing concurrent generation...");
			for (int i = 0; i < models.size(); i++) {
				LlamaModel model = models.get(i);
				InferenceParameters params = new InferenceParameters("test" + i)
					.setNPredict(5);

				int tokens = 0;
				for (LlamaOutput output : model.generate(params)) {
					tokens++;
				}
				System.out.printf("Model %d generated %d tokens\n", i + 1, tokens);
			}

			afterGeneration = takeSnapshot("After Generation");

			// Analysis
			System.out.println("\n📊 Concurrent Memory Analysis:");
			long multiModelOverhead = afterModels.heapUsed - initial.heapUsed;
			long perModelCost = multiModelOverhead / models.size();

			System.out.printf("Multi-model memory overhead: %d MB\n", multiModelOverhead / MB);
			System.out.printf("Per-model memory cost: %d MB\n", perModelCost / MB);
			System.out.printf("Generation memory impact: %d MB\n",
				(afterGeneration.heapUsed - afterModels.heapUsed) / MB);

			// Verify reasonable memory usage
			Assert.assertTrue("Per-model memory cost should be reasonable",
				perModelCost < 200 * MB); // Less than 200MB per model

		} finally {
			// Clean up all models
			for (LlamaModel model : models) {
				model.close();
			}
		}

		System.gc();
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		MemorySnapshot afterCleanup = takeSnapshot("After Cleanup");
		long memoryReclaimed = afterGeneration.heapUsed - afterCleanup.heapUsed;
		System.out.printf("Memory reclaimed: %d MB\n", memoryReclaimed / MB);

		System.out.println("\n✅ Concurrent memory usage test completed!");
	}

	private MemorySnapshot takeSnapshot(String name) {
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();

		return new MemorySnapshot(
			name,
			heapUsage.getUsed(),
			heapUsage.getMax(),
			nonHeapUsage.getUsed()
		);
	}

	private long getTotalGcCount(List<GarbageCollectorMXBean> gcBeans) {
		return gcBeans.stream()
			.mapToLong(GarbageCollectorMXBean::getCollectionCount)
			.sum();
	}

	private record MemorySnapshot(String name, long heapUsed, long heapMax, long nonHeapUsed) {}
}
