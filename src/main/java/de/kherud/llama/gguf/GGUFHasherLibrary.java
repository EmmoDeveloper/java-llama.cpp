package de.kherud.llama.gguf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Library-friendly GGUF file hashing utility.
 *
 * This refactored version provides a fluent API for calculating GGUF file hashes,
 * with builder pattern configuration, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic single file hashing
 * HashResult result = GGUFHasherLibrary.open(ggufPath)
 *     .algorithms(HashAlgorithm.SHA256, HashAlgorithm.MD5)
 *     .hash();
 *
 * // Multiple files with progress
 * List<HashResult> results = GGUFHasherLibrary.builder()
 *     .algorithms(EnumSet.allOf(HashAlgorithm.class))
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .parallel(true)
 *     .build()
 *     .hashFiles(Arrays.asList(file1, file2, file3));
 *
 * // Async single file
 * GGUFHasherLibrary.open(ggufPath)
 *     .algorithms(HashAlgorithm.SHA256)
 *     .hashAsync()
 *     .thenAccept(result -> System.out.println("SHA256: " + result.getHash(HashAlgorithm.SHA256)));
 *
 * // Directory hashing
 * DirectoryHashResult dirResult = GGUFHasherLibrary.builder()
 *     .algorithms(HashAlgorithm.SHA256, HashAlgorithm.GGUF_CONTENT)
 *     .build()
 *     .hashDirectory(Paths.get("models/"));
 * }</pre>
 */
public class GGUFHasherLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(GGUFHasherLibrary.class.getName());

	private final Set<GGUFHasher.HashAlgorithm> algorithms;
	private final boolean parallel;
	private final int bufferSize;
	private final Consumer<HashProgress> progressCallback;
	private final ExecutorService executor;

	private GGUFHasherLibrary(Builder builder) {
		this.algorithms = EnumSet.copyOf(builder.algorithms);
		this.parallel = builder.parallel;
		this.bufferSize = builder.bufferSize;
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static GGUFHasherLibrary open(Path filePath) {
		return builder().build();
	}

	// Fluent API for configuring algorithms
	public GGUFHasherLibrary algorithms(GGUFHasher.HashAlgorithm... algorithms) {
		this.algorithms.clear();
		Collections.addAll(this.algorithms, algorithms);
		return this;
	}

	public GGUFHasherLibrary algorithms(Set<GGUFHasher.HashAlgorithm> algorithms) {
		this.algorithms.clear();
		this.algorithms.addAll(algorithms);
		return this;
	}

	public GGUFHasherLibrary addAlgorithm(GGUFHasher.HashAlgorithm algorithm) {
		this.algorithms.add(algorithm);
		return this;
	}

	/**
	 * Hash a single file
	 */
	public HashResult hash(Path filePath) throws IOException {
		validateFilePath(filePath);

		progress("Starting hash calculation for " + filePath.getFileName(), 0.0);
		Instant startTime = Instant.now();

		try {
			// Use the original hasher for the actual work
			GGUFHasher.HashOptions options = new GGUFHasher.HashOptions()
				.algorithms(algorithms.toArray(new GGUFHasher.HashAlgorithm[0]))
				.bufferSize(bufferSize)
				.threads(parallel ? Runtime.getRuntime().availableProcessors() : 1);

			GGUFHasher hasher = new GGUFHasher(options);
			GGUFHasher.HashResult originalResult = hasher.hashFile(filePath);

			progress("Hash calculation complete", 1.0);

			// Convert to our result format
			return new HashResult.Builder()
				.success(!originalResult.hasError())
				.message(originalResult.hasError() ? originalResult.getError() : "Hash calculation successful")
				.filePath(filePath)
				.fileSize(originalResult.getFileSize())
				.hashes(originalResult.getHashes())
				.duration(Duration.between(startTime, Instant.now()))
				.algorithms(algorithms)
				.build();

		} catch (Exception e) {
			String errorMsg = "Hash calculation failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new HashResult.Builder()
				.success(false)
				.message(errorMsg)
				.filePath(filePath)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Hash a single file (for fluent API usage)
	 */
	public HashResult hash() throws IOException {
		throw new IllegalStateException("No file path specified. Use hash(Path) or configure with open(Path)");
	}

	/**
	 * Hash multiple files
	 */
	public List<HashResult> hashFiles(List<Path> filePaths) throws IOException {
		List<HashResult> results = new ArrayList<>();
		int totalFiles = filePaths.size();

		for (int i = 0; i < totalFiles; i++) {
			Path filePath = filePaths.get(i);
			progress("Processing file " + (i + 1) + "/" + totalFiles + ": " + filePath.getFileName(),
				(double) i / totalFiles);

			try {
				HashResult result = hash(filePath);
				results.add(result);
			} catch (IOException e) {
				results.add(new HashResult.Builder()
					.success(false)
					.message("Failed to hash file: " + e.getMessage())
					.filePath(filePath)
					.error(e)
					.build());
			}
		}

		progress("All files processed", 1.0);
		return results;
	}

	/**
	 * Hash all GGUF files in a directory
	 */
	public DirectoryHashResult hashDirectory(Path directoryPath) throws IOException {
		if (!Files.exists(directoryPath) || !Files.isDirectory(directoryPath)) {
			throw new IOException("Directory does not exist or is not a directory: " + directoryPath);
		}

		progress("Scanning directory for GGUF files", 0.1);

		// Find all GGUF files
		List<Path> ggufFiles = new ArrayList<>();
		Files.walk(directoryPath)
			.filter(Files::isRegularFile)
			.filter(path -> path.toString().toLowerCase().endsWith(".gguf"))
			.forEach(ggufFiles::add);

		progress("Found " + ggufFiles.size() + " GGUF files", 0.2);

		// Hash all files
		List<HashResult> fileResults = hashFiles(ggufFiles);

		// Calculate directory-level statistics
		int successCount = 0;
		int errorCount = 0;
		long totalSize = 0;
		Duration totalDuration = Duration.ZERO;

		for (HashResult result : fileResults) {
			if (result.isSuccess()) {
				successCount++;
				totalSize += result.getFileSize();
			} else {
				errorCount++;
			}
			totalDuration = totalDuration.plus(result.getDuration());
		}

		return new DirectoryHashResult.Builder()
			.directoryPath(directoryPath)
			.fileResults(fileResults)
			.totalFiles(ggufFiles.size())
			.successCount(successCount)
			.errorCount(errorCount)
			.totalSize(totalSize)
			.totalDuration(totalDuration)
			.algorithms(algorithms)
			.build();
	}

	/**
	 * Hash a file asynchronously
	 */
	public CompletableFuture<HashResult> hashAsync(Path filePath) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();

		return CompletableFuture.supplyAsync(() -> {
			try {
				return hash(filePath);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}, exec);
	}

	/**
	 * Hash multiple files asynchronously
	 */
	public CompletableFuture<List<HashResult>> hashFilesAsync(List<Path> filePaths) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();

		return CompletableFuture.supplyAsync(() -> {
			try {
				return hashFiles(filePaths);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}, exec);
	}

	/**
	 * Verify file integrity using existing hash
	 */
	public VerificationResult verify(Path filePath, GGUFHasher.HashAlgorithm algorithm, String expectedHash) throws IOException {
		HashResult result = hash(filePath);

		if (!result.isSuccess()) {
			return new VerificationResult(filePath, algorithm, expectedHash, null, false, result.getMessage());
		}

		String actualHash = result.getHash(algorithm);
		boolean matches = expectedHash.equalsIgnoreCase(actualHash);

		return new VerificationResult(filePath, algorithm, expectedHash, actualHash, matches,
			matches ? "Hash verification successful" : "Hash mismatch detected");
	}

	// Helper methods
	private void validateFilePath(Path filePath) throws IOException {
		if (!Files.exists(filePath)) {
			throw new IOException("File does not exist: " + filePath);
		}
		if (!Files.isRegularFile(filePath)) {
			throw new IOException("Path is not a regular file: " + filePath);
		}
		if (!Files.isReadable(filePath)) {
			throw new IOException("File is not readable: " + filePath);
		}
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new HashProgress(message, progress));
		}
	}

	@Override
	public void close() throws IOException {
		if (executor != null) {
			executor.shutdown();
		}
	}

	// Builder class
	public static class Builder {
		private Set<GGUFHasher.HashAlgorithm> algorithms = EnumSet.of(GGUFHasher.HashAlgorithm.SHA256);
		private boolean parallel = true;
		private int bufferSize = 8192;
		private Consumer<HashProgress> progressCallback;
		private ExecutorService executor;

		public Builder algorithms(GGUFHasher.HashAlgorithm... algorithms) {
			this.algorithms = EnumSet.noneOf(GGUFHasher.HashAlgorithm.class);
			Collections.addAll(this.algorithms, algorithms);
			return this;
		}

		public Builder algorithms(Set<GGUFHasher.HashAlgorithm> algorithms) {
			this.algorithms = EnumSet.copyOf(algorithms);
			return this;
		}

		public Builder parallel(boolean parallel) {
			this.parallel = parallel;
			return this;
		}

		public Builder bufferSize(int bufferSize) {
			this.bufferSize = Math.max(1024, bufferSize);
			return this;
		}

		public Builder progressCallback(Consumer<HashProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}


		public GGUFHasherLibrary build() {
			return new GGUFHasherLibrary(this);
		}
	}

	// Progress tracking class
	public static class HashProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public HashProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	// Enhanced result class
	public static class HashResult {
		private final boolean success;
		private final String message;
		private final Path filePath;
		private final long fileSize;
		private final Map<GGUFHasher.HashAlgorithm, String> hashes;
		private final Duration duration;
		private final Set<GGUFHasher.HashAlgorithm> algorithms;
		private final Exception error;

		private HashResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.filePath = builder.filePath;
			this.fileSize = builder.fileSize;
			this.hashes = Collections.unmodifiableMap(builder.hashes);
			this.duration = builder.duration;
			this.algorithms = Collections.unmodifiableSet(builder.algorithms);
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Path getFilePath() { return filePath; }
		public long getFileSize() { return fileSize; }
		public Map<GGUFHasher.HashAlgorithm, String> getHashes() { return hashes; }
		public String getHash(GGUFHasher.HashAlgorithm algorithm) { return hashes.get(algorithm); }
		public Duration getDuration() { return duration; }
		public Set<GGUFHasher.HashAlgorithm> getAlgorithms() { return algorithms; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public boolean hasHash(GGUFHasher.HashAlgorithm algorithm) {
			return hashes.containsKey(algorithm);
		}

		public static class Builder {
			private boolean success;
			private String message;
			private Path filePath;
			private long fileSize;
			private Map<GGUFHasher.HashAlgorithm, String> hashes = new LinkedHashMap<>();
			private Duration duration = Duration.ZERO;
			private Set<GGUFHasher.HashAlgorithm> algorithms = EnumSet.noneOf(GGUFHasher.HashAlgorithm.class);
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder filePath(Path filePath) { this.filePath = filePath; return this; }
			public Builder fileSize(long fileSize) { this.fileSize = fileSize; return this; }
			public Builder hashes(Map<GGUFHasher.HashAlgorithm, String> hashes) { this.hashes = hashes; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder algorithms(Set<GGUFHasher.HashAlgorithm> algorithms) { this.algorithms = algorithms; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public HashResult build() { return new HashResult(this); }
		}
	}

	// Directory hash result class
	public static class DirectoryHashResult {
		private final Path directoryPath;
		private final List<HashResult> fileResults;
		private final int totalFiles;
		private final int successCount;
		private final int errorCount;
		private final long totalSize;
		private final Duration totalDuration;
		private final Set<GGUFHasher.HashAlgorithm> algorithms;

		private DirectoryHashResult(Builder builder) {
			this.directoryPath = builder.directoryPath;
			this.fileResults = Collections.unmodifiableList(builder.fileResults);
			this.totalFiles = builder.totalFiles;
			this.successCount = builder.successCount;
			this.errorCount = builder.errorCount;
			this.totalSize = builder.totalSize;
			this.totalDuration = builder.totalDuration;
			this.algorithms = Collections.unmodifiableSet(builder.algorithms);
		}

		public Path getDirectoryPath() { return directoryPath; }
		public List<HashResult> getFileResults() { return fileResults; }
		public int getTotalFiles() { return totalFiles; }
		public int getSuccessCount() { return successCount; }
		public int getErrorCount() { return errorCount; }
		public long getTotalSize() { return totalSize; }
		public Duration getTotalDuration() { return totalDuration; }
		public Set<GGUFHasher.HashAlgorithm> getAlgorithms() { return algorithms; }

		public List<HashResult> getSuccessfulResults() {
			return fileResults.stream().filter(HashResult::isSuccess).toList();
		}

		public List<HashResult> getFailedResults() {
			return fileResults.stream().filter(result -> !result.isSuccess()).toList();
		}

		public static class Builder {
			private Path directoryPath;
			private List<HashResult> fileResults = new ArrayList<>();
			private int totalFiles;
			private int successCount;
			private int errorCount;
			private long totalSize;
			private Duration totalDuration = Duration.ZERO;
			private Set<GGUFHasher.HashAlgorithm> algorithms = EnumSet.noneOf(GGUFHasher.HashAlgorithm.class);

			public Builder directoryPath(Path directoryPath) { this.directoryPath = directoryPath; return this; }
			public Builder fileResults(List<HashResult> fileResults) { this.fileResults = fileResults; return this; }
			public Builder totalFiles(int totalFiles) { this.totalFiles = totalFiles; return this; }
			public Builder successCount(int successCount) { this.successCount = successCount; return this; }
			public Builder errorCount(int errorCount) { this.errorCount = errorCount; return this; }
			public Builder totalSize(long totalSize) { this.totalSize = totalSize; return this; }
			public Builder totalDuration(Duration totalDuration) { this.totalDuration = totalDuration; return this; }
			public Builder algorithms(Set<GGUFHasher.HashAlgorithm> algorithms) { this.algorithms = algorithms; return this; }

			public DirectoryHashResult build() { return new DirectoryHashResult(this); }
		}
	}

	// Verification result class
	public static class VerificationResult {
		private final Path filePath;
		private final GGUFHasher.HashAlgorithm algorithm;
		private final String expectedHash;
		private final String actualHash;
		private final boolean matches;
		private final String message;

		public VerificationResult(Path filePath, GGUFHasher.HashAlgorithm algorithm, String expectedHash, String actualHash, boolean matches, String message) {
			this.filePath = filePath;
			this.algorithm = algorithm;
			this.expectedHash = expectedHash;
			this.actualHash = actualHash;
			this.matches = matches;
			this.message = message;
		}

		public Path getFilePath() { return filePath; }
		public GGUFHasher.HashAlgorithm getAlgorithm() { return algorithm; }
		public String getExpectedHash() { return expectedHash; }
		public String getActualHash() { return actualHash; }
		public boolean matches() { return matches; }
		public String getMessage() { return message; }
	}
}