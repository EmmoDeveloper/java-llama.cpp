package de.kherud.llama;

import de.kherud.llama.util.ConsoleCancellationHandler;
import org.junit.Assert;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class CancellationTest {

	@Test
	public void testIteratorCancellation() throws Exception {
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);

		try (LlamaModel model = new LlamaModel(params)) {
			// Generate with many tokens to ensure we can cancel mid-generation
			InferenceParameters inferParams = new InferenceParameters("Count from 1 to 100:")
				.setNPredict(200)
				.setStream(true);

			LlamaIterator iterator = model.generate(inferParams).iterator();
			AtomicInteger tokenCount = new AtomicInteger(0);

			// Process a few tokens then cancel
			while (iterator.hasNext() && tokenCount.get() < 5) {
				iterator.next();
				tokenCount.incrementAndGet();
			}

			// Cancel the iterator
			iterator.cancel();

			// Verify iteration stops after cancellation
			Assert.assertFalse("Iterator should not have next after cancel", iterator.hasNext());
			Assert.assertTrue("Should have generated some tokens before cancel", tokenCount.get() > 0);
			Assert.assertTrue("Should have cancelled before completing all tokens", tokenCount.get() < 50);
		}
	}

	@Test
	public void testCancellationHandlerMonitoring() throws Exception {
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);

		try (LlamaModel model = new LlamaModel(params)) {
			InferenceParameters inferParams = new InferenceParameters("Hello")
				.setNPredict(10)
				.setStream(true);

			LlamaIterator iterator = model.generate(inferParams).iterator();
			ConsoleCancellationHandler handler = new ConsoleCancellationHandler();

			// Start listening
			handler.listen(iterator);

			// Process tokens
			int count = 0;
			while (iterator.hasNext() && count < 5) {
				iterator.next();
				count++;
			}

			// Stop listening
			handler.stop();

			// Handler should not report cancelled if we didn't press ESC
			Assert.assertFalse("Should not be cancelled without ESC press", handler.wasCancelled());
		}
	}

	@Test
	public void testProcessWithCancellation() throws Exception {
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);

		try (LlamaModel model = new LlamaModel(params)) {
			InferenceParameters inferParams = new InferenceParameters("Test")
				.setNPredict(20)
				.setStream(true);

			LlamaIterator iterator = model.generate(inferParams).iterator();
			AtomicInteger processedCount = new AtomicInteger(0);
			CountDownLatch latch = new CountDownLatch(1);

			// Process in a separate thread so we can control timing
			Thread processingThread = new Thread(() -> {
				ConsoleCancellationHandler.processWithCancellation(iterator, output -> {
					processedCount.incrementAndGet();
					// Simulate cancellation after a few tokens
					if (processedCount.get() >= 3) {
						iterator.cancel();
					}
				});
				latch.countDown();
			});

			processingThread.start();

			// Wait for processing to complete
			Assert.assertTrue("Processing should complete", latch.await(10, TimeUnit.SECONDS));

			// Verify some tokens were processed
			Assert.assertTrue("Should have processed some tokens", processedCount.get() > 0);
			Assert.assertTrue("Should have stopped early due to cancellation", processedCount.get() < 10);
		}
	}

	@Test
	public void testCancellationHandlerNullCheck() {
		ConsoleCancellationHandler handler = new ConsoleCancellationHandler();

		try {
			handler.listen(null);
			Assert.fail("Should throw IllegalArgumentException for null iterator");
		} catch (IllegalArgumentException e) {
			// Expected
			Assert.assertEquals("Iterator cannot be null", e.getMessage());
		}
	}
}
