package de.kherud.llama;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Async wrapper for LlamaModel providing non-blocking operations.
 * Manages thread pool for concurrent inference requests.
 */
public class LlamaAsyncService implements AutoCloseable {

	private final LlamaModel model;
	private final ExecutorService executor;
	private final boolean ownsModel;

	/**
	 * Create async service with a new model instance
	 */
	public LlamaAsyncService(ModelParameters parameters) {
		this(new LlamaModel(parameters), true);
	}

	/**
	 * Create async service wrapping an existing model
	 */
	public LlamaAsyncService(LlamaModel model) {
		this(model, false);
	}

	private LlamaAsyncService(LlamaModel model, boolean ownsModel) {
		this.model = model;
		this.ownsModel = ownsModel;
		this.executor = createExecutor();
	}

	private static ExecutorService createExecutor() {
		ThreadFactory threadFactory = new ThreadFactory() {
			private final AtomicInteger counter = new AtomicInteger(0);
			@Override
			public Thread newThread(Runnable r) {
				Thread thread = new Thread(r);
				thread.setName("llama-async-" + counter.getAndIncrement());
				thread.setDaemon(true);
				return thread;
			}
		};

		int threadCount = Math.min(4, Runtime.getRuntime().availableProcessors());
		return Executors.newFixedThreadPool(threadCount, threadFactory);
	}

	/**
	 * Asynchronously generate a complete response
	 */
	public CompletableFuture<String> completeAsync(InferenceParameters parameters) {
		return CompletableFuture.supplyAsync(() -> model.complete(parameters), executor);
	}

	/**
	 * Asynchronously generate a complete response with prompt
	 */
	public CompletableFuture<String> completeAsync(String prompt) {
		return completeAsync(new InferenceParameters(prompt));
	}

	/**
	 * Asynchronously start streaming generation
	 */
	public CompletableFuture<Stream<LlamaOutput>> generateAsync(InferenceParameters parameters) {
		return CompletableFuture.supplyAsync(() -> {
			Iterable<LlamaOutput> iterable = model.generate(parameters);
			return StreamSupport.stream(iterable.spliterator(), false);
		}, executor);
	}

	/**
	 * Asynchronously start streaming generation with prompt
	 */
	public CompletableFuture<Stream<LlamaOutput>> generateAsync(String prompt) {
		return generateAsync(new InferenceParameters(prompt));
	}

	/**
	 * Asynchronously generate embeddings
	 */
	public CompletableFuture<float[]> embedAsync(String text) {
		return CompletableFuture.supplyAsync(() -> model.embed(text), executor);
	}

	/**
	 * Asynchronously tokenize text
	 */
	public CompletableFuture<int[]> encodeAsync(String text) {
		return CompletableFuture.supplyAsync(() -> model.encode(text), executor);
	}

	/**
	 * Asynchronously decode tokens
	 */
	public CompletableFuture<String> decodeAsync(int[] tokens) {
		return CompletableFuture.supplyAsync(() -> model.decode(tokens), executor);
	}

	/**
	 * Get the underlying model (for synchronous operations)
	 */
	public LlamaModel getModel() {
		return model;
	}

	/**
	 * Check if service is ready
	 */
	public boolean isReady() {
		return model != null && !executor.isShutdown();
	}

	@Override
	public void close() {
		executor.shutdown();
		if (ownsModel && model != null) {
			try {
				model.close();
			} catch (Exception e) {
				// Log error
			}
		}
	}
}
