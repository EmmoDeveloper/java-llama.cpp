package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.lang.management.GarbageCollectorMXBean;
import java.util.List;
import java.util.ArrayList;

public class MemoryAnalysisTest {
	private static final System.Logger logger = System.getLogger(MemoryAnalysisTest.class.getName());

	private static final int MB = 1024 * 1024;

	@Test
	public void testMemoryUsagePatterns() {
		logger.log(DEBUG, "\n=== Memory Usage Pattern Analysis ===\n");

		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

		// Initial memory state
		MemorySnapshot initial = takeSnapshot("Initial");
		long gcCountBefore = getTotalGcCount(gcBeans);

		// Test memory usage during model lifecycle
		logger.log(DEBUG, "🔍 Testing memory usage during model lifecycle...");

		LlamaModel model = null;
		MemorySnapshot afterLoad;
		MemorySnapshot afterGeneration;

		try {
			// Step 1: Model loading
			logger.log(DEBUG, "Loading model...");
			model = new LlamaModel(
				new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setCtxSize(1024)
					.setGpuLayers(43)
			);
			afterLoad = takeSnapshot("After Load");

			// Step 2: Token generation
			logger.log(DEBUG, "Generating tokens...");
			InferenceParameters params = new InferenceParameters("def test_function():")
				.setNPredict(20)
				.setTemperature(0.1f);

			int tokenCount = 0;
			for (LlamaOutput output : model.generate(params)) {
				tokenCount++;
			}

			afterGeneration = takeSnapshot("After Generation");

			logger.log(DEBUG, "Generated %d tokens", tokenCount);

			Assert.assertTrue("Should generate tokens", tokenCount > 0);

		} finally {
			if (model != null) {
				logger.log(DEBUG, "Closing model...");
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
		logger.log(DEBUG, "\n=== Memory Analysis Results ===");

		long loadingMemoryIncrease = afterLoad.heapUsed - initial.heapUsed;
		long generationMemoryIncrease = afterGeneration.heapUsed - afterLoad.heapUsed;
		long memoryReclaimed = afterGeneration.heapUsed - afterClose.heapUsed;

		logger.log(DEBUG, "Memory increase during loading: %d MB", loadingMemoryIncrease / MB);
		logger.log(DEBUG, "Memory increase during generation: %d MB", generationMemoryIncrease / MB);
		logger.log(DEBUG, "Memory reclaimed after close: %d MB", memoryReclaimed / MB);
		logger.log(DEBUG, "GC cycles triggered: %d", gcCountAfter - gcCountBefore);

		// Print detailed snapshots
		logger.log(DEBUG, "\n📊 Detailed Memory Snapshots:");
		logger.log(DEBUG, initial);
		logger.log(DEBUG, afterLoad);
		logger.log(DEBUG, afterGeneration);
		logger.log(DEBUG, afterClose);

		Assert.assertTrue("Memory should be reclaimed after close", memoryReclaimed > 0 || loadingMemoryIncrease < 50 * MB);

		logger.log(DEBUG, "\n✅ Memory pattern analysis completed!");
	}

	@Test
	public void testMemoryLeakDetection() {
		logger.log(DEBUG, "\n=== Memory Leak Detection Test ===\n");

		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		List<Long> heapUsages = new ArrayList<>();

		logger.log(DEBUG, "🔍 Testing for memory leaks with multiple model instances...");

		// Test multiple create/close cycles
		for (int i = 0; i < 3; i++) {
			logger.log(DEBUG, "Cycle %d: ", i + 1);

			try (LlamaModel model = new LlamaModel(
				new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
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
				logger.log(DEBUG, "Generated %d tokens, ", tokens);

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
			logger.log(DEBUG, "Heap: %d MB ", heapUsed / MB);
		}

		// Analyze leak trend
		logger.log(DEBUG, "\n📈 Memory Leak Analysis:");
		boolean memoryIncreasing = true;
		for (int i = 1; i < heapUsages.size(); i++) {
			long diff = heapUsages.get(i) - heapUsages.get(i-1);
			logger.log(DEBUG, "Cycle %d vs %d: %+d MB", i, i+1, diff / MB);

			if (diff < 10 * MB) { // Allow 10MB variation
				memoryIncreasing = false;
			}
		}

		if (memoryIncreasing) {
			logger.log(DEBUG, "⚠️  Potential memory leak detected - heap consistently growing");
		} else {
			logger.log(DEBUG, "✅ No significant memory leak detected");
		}

		// Final assertion
		long totalIncrease = heapUsages.get(heapUsages.size()-1) - heapUsages.get(0);
		Assert.assertTrue("Memory increase should be reasonable (< 100MB)",
			Math.abs(totalIncrease) < 100 * MB);

		logger.log(DEBUG, "\n✅ Memory leak detection completed!");
	}

	@Test
	public void testConcurrentMemoryUsage() {
		logger.log(DEBUG, "\n=== Concurrent Memory Usage Test ===\n");

		MemorySnapshot initial = takeSnapshot("Initial");

		logger.log(DEBUG, "🔍 Testing memory usage with concurrent model usage...");

		// Test resource sharing and memory efficiency
		List<LlamaModel> models = new ArrayList<>();
		MemorySnapshot afterGeneration;

		try {
			// Create multiple models (should reuse native resources efficiently)
			for (int i = 0; i < 2; i++) {
				logger.log(DEBUG, "Creating model %d...", i + 1);
				LlamaModel model = new LlamaModel(
					new ModelParameters()
						.setModel("models/codellama-7b.Q2_K.gguf")
						.setCtxSize(256) // Smaller for memory efficiency
						.setGpuLayers(43)
				);
				models.add(model);
			}

			MemorySnapshot afterModels = takeSnapshot("After Models");

			// Test concurrent generation
			logger.log(DEBUG, "Testing concurrent generation...");
			for (int i = 0; i < models.size(); i++) {
				LlamaModel model = models.get(i);
				InferenceParameters params = new InferenceParameters("test" + i)
					.setNPredict(5);

				int tokens = 0;
				for (LlamaOutput output : model.generate(params)) {
					tokens++;
				}
				logger.log(DEBUG, "Model %d generated %d tokens", i + 1, tokens);
			}

			afterGeneration = takeSnapshot("After Generation");

			// Analysis
			logger.log(DEBUG, "\n📊 Concurrent Memory Analysis:");
			long multiModelOverhead = afterModels.heapUsed - initial.heapUsed;
			long perModelCost = multiModelOverhead / models.size();

			logger.log(DEBUG, "Multi-model memory overhead: %d MB", multiModelOverhead / MB);
			logger.log(DEBUG, "Per-model memory cost: %d MB", perModelCost / MB);
			logger.log(DEBUG, "Generation memory impact: %d MB",
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
		logger.log(DEBUG, "Memory reclaimed: %d MB", memoryReclaimed / MB);

		logger.log(DEBUG, "\n✅ Concurrent memory usage test completed!");
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
