package de.kherud.llama.performance;

import java.util.concurrent.*;
import java.util.logging.Logger;
import java.util.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Performance optimization utilities for heavy operations.
 *
 * Provides optimized implementations for file hashing, validation,
 * and other computationally intensive tasks used by the utility classes.
 */
public class PerformanceOptimizer {
	private static final Logger LOGGER = Logger.getLogger(PerformanceOptimizer.class.getName());

	// Optimal buffer sizes for different operations
	private static final int HASH_BUFFER_SIZE = 1024 * 1024; // 1MB for hashing
	private static final int VALIDATION_BUFFER_SIZE = 512 * 1024; // 512KB for validation
	private static final int IO_BUFFER_SIZE = 64 * 1024; // 64KB for general I/O

	// Thread pool for parallel operations
	private static final ExecutorService HASH_POOL = Executors.newFixedThreadPool(
		Math.max(2, Runtime.getRuntime().availableProcessors() / 2),
		r -> {
			Thread t = new Thread(r, "hash-worker");
			t.setDaemon(true);
			return t;
		}
	);

	/**
	 * Optimized file hashing using memory-mapped files and parallel processing
	 */
	public static class OptimizedHasher {
		private final String algorithm;
		private final int bufferSize;
		private final boolean useMemoryMapping;

		public OptimizedHasher(String algorithm) {
			this(algorithm, HASH_BUFFER_SIZE, true);
		}

		public OptimizedHasher(String algorithm, int bufferSize, boolean useMemoryMapping) {
			this.algorithm = algorithm;
			this.bufferSize = bufferSize;
			this.useMemoryMapping = useMemoryMapping;
		}

		/**
		 * Compute hash using optimized approach
		 */
		public String computeHash(Path filePath) throws IOException, NoSuchAlgorithmException {
			long fileSize = java.nio.file.Files.size(filePath);

			// Choose strategy based on file size
			if (useMemoryMapping && fileSize < Integer.MAX_VALUE && fileSize > bufferSize * 4) {
				return computeHashMemoryMapped(filePath);
			} else {
				return computeHashBuffered(filePath);
			}
		}

		/**
		 * Memory-mapped file hashing for large files
		 */
		private String computeHashMemoryMapped(Path filePath) throws IOException, NoSuchAlgorithmException {
			MessageDigest digest = MessageDigest.getInstance(algorithm);

			try (FileChannel channel = FileChannel.open(filePath, StandardOpenOption.READ)) {
				long fileSize = channel.size();
				long position = 0;

				// Process file in chunks to avoid memory issues
				while (position < fileSize) {
					long chunkSize = Math.min(bufferSize, fileSize - position);
					ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, position, chunkSize);

					digest.update(buffer);
					position += chunkSize;
				}
			}

			return bytesToHex(digest.digest());
		}

		/**
		 * Buffered file hashing for smaller files
		 */
		private String computeHashBuffered(Path filePath) throws IOException, NoSuchAlgorithmException {
			MessageDigest digest = MessageDigest.getInstance(algorithm);
			ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize);

			try (FileChannel channel = FileChannel.open(filePath, StandardOpenOption.READ)) {
				while (channel.read(buffer) > 0) {
					buffer.flip();
					digest.update(buffer);
					buffer.clear();
				}
			}

			return bytesToHex(digest.digest());
		}

		/**
		 * Parallel hashing of multiple files
		 */
		public CompletableFuture<String> computeHashAsync(Path filePath) {
			return CompletableFuture.supplyAsync(() -> {
				try {
					return computeHash(filePath);
				} catch (Exception e) {
					throw new RuntimeException("Hash computation failed for: " + filePath, e);
				}
			}, HASH_POOL);
		}
	}

	/**
	 * Optimized file comparison for model validation
	 */
	public static class OptimizedComparator {
		private final int bufferSize;
		private final float tolerance;

		public OptimizedComparator() {
			this(VALIDATION_BUFFER_SIZE, 1e-6f);
		}

		public OptimizedComparator(int bufferSize, float tolerance) {
			this.bufferSize = bufferSize;
			this.tolerance = tolerance;
		}

		/**
		 * Fast binary file comparison
		 */
		public boolean filesEqual(Path file1, Path file2) throws IOException {
			// Quick size check first
			long size1 = java.nio.file.Files.size(file1);
			long size2 = java.nio.file.Files.size(file2);

			if (size1 != size2) {
				return false;
			}

			// Use memory-mapped comparison for large files
			if (size1 > bufferSize * 4) {
				return filesEqualMemoryMapped(file1, file2);
			} else {
				return filesEqualBuffered(file1, file2);
			}
		}

		private boolean filesEqualMemoryMapped(Path file1, Path file2) throws IOException {
			try (FileChannel channel1 = FileChannel.open(file1, StandardOpenOption.READ);
				 FileChannel channel2 = FileChannel.open(file2, StandardOpenOption.READ)) {

				long fileSize = channel1.size();
				long position = 0;

				while (position < fileSize) {
					long chunkSize = Math.min(bufferSize, fileSize - position);

					ByteBuffer buffer1 = channel1.map(FileChannel.MapMode.READ_ONLY, position, chunkSize);
					ByteBuffer buffer2 = channel2.map(FileChannel.MapMode.READ_ONLY, position, chunkSize);

					if (!buffer1.equals(buffer2)) {
						return false;
					}

					position += chunkSize;
				}
			}

			return true;
		}

		private boolean filesEqualBuffered(Path file1, Path file2) throws IOException {
			ByteBuffer buffer1 = ByteBuffer.allocateDirect(bufferSize);
			ByteBuffer buffer2 = ByteBuffer.allocateDirect(bufferSize);

			try (FileChannel channel1 = FileChannel.open(file1, StandardOpenOption.READ);
				 FileChannel channel2 = FileChannel.open(file2, StandardOpenOption.READ)) {

				while (true) {
					buffer1.clear();
					buffer2.clear();

					int read1 = channel1.read(buffer1);
					int read2 = channel2.read(buffer2);

					if (read1 != read2) {
						return false;
					}

					if (read1 == -1) {
						break; // End of both files
					}

					buffer1.flip();
					buffer2.flip();

					if (!buffer1.equals(buffer2)) {
						return false;
					}
				}
			}

			return true;
		}

		/**
		 * Compare floating-point arrays with tolerance
		 */
		public boolean arraysEqual(float[] array1, float[] array2) {
			if (array1.length != array2.length) {
				return false;
			}

			for (int i = 0; i < array1.length; i++) {
				if (Math.abs(array1[i] - array2[i]) > tolerance) {
					return false;
				}
			}

			return true;
		}

		/**
		 * Compute NMSE (Normalized Mean Square Error) efficiently
		 */
		public double computeNMSE(float[] reference, float[] test) {
			if (reference.length != test.length) {
				throw new IllegalArgumentException("Arrays must have the same length");
			}

			double sumSquaredError = 0.0;
			double sumSquaredReference = 0.0;

			// Vectorized computation
			for (int i = 0; i < reference.length; i++) {
				double error = test[i] - reference[i];
				sumSquaredError += error * error;
				sumSquaredReference += reference[i] * reference[i];
			}

			return sumSquaredReference > 0 ? sumSquaredError / sumSquaredReference : 0.0;
		}
	}

	/**
	 * Memory-efficient tensor processing
	 */
	public static class TensorProcessor {
		private final int chunkSize;

		public TensorProcessor() {
			this(1024 * 1024); // 1M elements per chunk
		}

		public TensorProcessor(int chunkSize) {
			this.chunkSize = chunkSize;
		}

		/**
		 * Process large tensors in chunks to avoid memory issues
		 */
		public void processTensorChunked(float[] tensor, TensorOperation operation) {
			int totalElements = tensor.length;
			int processed = 0;

			while (processed < totalElements) {
				int chunkEnd = Math.min(processed + chunkSize, totalElements);
				float[] chunk = java.util.Arrays.copyOfRange(tensor, processed, chunkEnd);

				operation.process(chunk, processed);

				processed = chunkEnd;
			}
		}

		/**
		 * Parallel tensor processing
		 */
		public void processTensorParallel(float[] tensor, TensorOperation operation) {
			int numThreads = Math.min(4, Runtime.getRuntime().availableProcessors());
			int elementsPerThread = tensor.length / numThreads;

			List<CompletableFuture<Void>> futures = new ArrayList<>();

			for (int i = 0; i < numThreads; i++) {
				final int threadIndex = i;
				final int start = i * elementsPerThread;
				final int end = (i == numThreads - 1) ? tensor.length : (i + 1) * elementsPerThread;

				CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
					float[] chunk = java.util.Arrays.copyOfRange(tensor, start, end);
					operation.process(chunk, start);
				}, HASH_POOL);

				futures.add(future);
			}

			// Wait for all threads to complete
			CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
		}

		@FunctionalInterface
		public interface TensorOperation {
			void process(float[] chunk, int offset);
		}
	}

	/**
	 * Optimized I/O operations
	 */
	public static class OptimizedIO {

		/**
		 * Fast file copying using NIO
		 */
		public static void copyFile(Path source, Path target) throws IOException {
			try (FileChannel sourceChannel = FileChannel.open(source, StandardOpenOption.READ);
				 FileChannel targetChannel = FileChannel.open(target,
					 StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {

				long size = sourceChannel.size();
				long position = 0;

				while (position < size) {
					long transferred = sourceChannel.transferTo(position, size - position, targetChannel);
					position += transferred;
				}
			}
		}

		/**
		 * Efficient file reading with optimal buffer size
		 */
		public static ByteBuffer readFileOptimized(Path filePath) throws IOException {
			long fileSize = java.nio.file.Files.size(filePath);

			if (fileSize > Integer.MAX_VALUE) {
				throw new IOException("File too large for ByteBuffer: " + fileSize);
			}

			ByteBuffer buffer = ByteBuffer.allocateDirect((int) fileSize);

			try (FileChannel channel = FileChannel.open(filePath, StandardOpenOption.READ)) {
				while (buffer.hasRemaining()) {
					if (channel.read(buffer) == -1) {
						break;
					}
				}
			}

			buffer.flip();
			return buffer;
		}
	}

	/**
	 * CPU-aware optimization
	 */
	public static class CPUOptimizer {
		private static final int CPU_CORES = Runtime.getRuntime().availableProcessors();

		/**
		 * Get optimal thread count for CPU-intensive operations
		 */
		public static int getOptimalThreadCount() {
			return Math.max(1, CPU_CORES - 1); // Leave one core for OS
		}

		/**
		 * Get optimal thread count for I/O operations
		 */
		public static int getOptimalIOThreadCount() {
			return Math.min(8, CPU_CORES * 2); // I/O can benefit from more threads
		}

		/**
		 * Get optimal buffer size based on available memory
		 */
		public static int getOptimalBufferSize() {
			Runtime runtime = Runtime.getRuntime();
			long maxMemory = runtime.maxMemory();
			long freeMemory = runtime.freeMemory();
			long availableMemory = maxMemory - (runtime.totalMemory() - freeMemory);

			// Use up to 10% of available memory for buffers, min 64KB, max 16MB
			long bufferSize = Math.max(64 * 1024, Math.min(16 * 1024 * 1024, availableMemory / 10));
			return (int) bufferSize;
		}
	}

	/**
	 * Cache for frequently accessed data
	 */
	public static class PerformanceCache {
		private static final int MAX_CACHE_SIZE = 100;
		private static final Map<String, Object> cache = new ConcurrentHashMap<>();
		private static final Map<String, Long> accessTimes = new ConcurrentHashMap<>();

		/**
		 * Get from cache with automatic cleanup
		 */
		@SuppressWarnings("unchecked")
		public static <T> T get(String key, Class<T> type) {
			accessTimes.put(key, System.currentTimeMillis());
			Object value = cache.get(key);

			// Cleanup old entries if cache is getting large
			if (cache.size() > MAX_CACHE_SIZE) {
				cleanupOldEntries();
			}

			return type.isInstance(value) ? type.cast(value) : null;
		}

		/**
		 * Put in cache
		 */
		public static void put(String key, Object value) {
			cache.put(key, value);
			accessTimes.put(key, System.currentTimeMillis());
		}

		/**
		 * Check if key exists in cache
		 */
		public static boolean containsKey(String key) {
			return cache.containsKey(key);
		}

		/**
		 * Clear cache
		 */
		public static void clear() {
			cache.clear();
			accessTimes.clear();
		}

		private static void cleanupOldEntries() {
			long cutoffTime = System.currentTimeMillis() - 300000; // 5 minutes

			accessTimes.entrySet().removeIf(entry -> {
				if (entry.getValue() < cutoffTime) {
					cache.remove(entry.getKey());
					return true;
				}
				return false;
			});
		}
	}

	/**
	 * Utility methods
	 */
	private static String bytesToHex(byte[] bytes) {
		StringBuilder result = new StringBuilder();
		for (byte b : bytes) {
			result.append(String.format("%02x", b));
		}
		return result.toString();
	}

	/**
	 * Shutdown performance-related resources
	 */
	public static void shutdown() {
		HASH_POOL.shutdown();
		try {
			if (!HASH_POOL.awaitTermination(10, TimeUnit.SECONDS)) {
				HASH_POOL.shutdownNow();
			}
		} catch (InterruptedException e) {
			HASH_POOL.shutdownNow();
			Thread.currentThread().interrupt();
		}
	}
}