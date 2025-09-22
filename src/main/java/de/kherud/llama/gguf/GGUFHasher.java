package de.kherud.llama.gguf;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

/**
 * GGUF file hashing utility.
 *
 * Equivalent to gguf_hash.py - calculates various checksums for GGUF files
 * including SHA-256, MD5, and custom GGUF-specific hashes.
 */
public class GGUFHasher {
	private static final Logger LOGGER = Logger.getLogger(GGUFHasher.class.getName());

	public enum HashAlgorithm {
		SHA256("SHA-256"),
		SHA1("SHA-1"),
		MD5("MD5"),
		GGUF_CONTENT("GGUF-Content"), // Hash only tensor data, not metadata
		GGUF_METADATA("GGUF-Metadata"); // Hash only metadata, not tensor data

		private final String algorithmName;

		HashAlgorithm(String algorithmName) {
			this.algorithmName = algorithmName;
		}

		public String getAlgorithmName() {
			return algorithmName;
		}
	}

	public static class HashOptions {
		private Set<HashAlgorithm> algorithms = EnumSet.of(HashAlgorithm.SHA256);
		private boolean recursive = false;
		private boolean verbose = false;
		private boolean outputFile = false;
		private String outputPath = null;
		private int bufferSize = 16 * 1024 * 1024; // 16MB buffer
		private int threads = Runtime.getRuntime().availableProcessors();

		public HashOptions algorithms(HashAlgorithm... algorithms) {
			this.algorithms = EnumSet.copyOf(Arrays.asList(algorithms));
			return this;
		}

		public HashOptions recursive(boolean recursive) {
			this.recursive = recursive;
			return this;
		}

		public HashOptions verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public HashOptions outputFile(String path) {
			this.outputFile = true;
			this.outputPath = path;
			return this;
		}

		public HashOptions bufferSize(int size) {
			this.bufferSize = size;
			return this;
		}

		public HashOptions threads(int threads) {
			this.threads = threads;
			return this;
		}
	}

	public static class HashResult {
		private final Path filePath;
		private final Map<HashAlgorithm, String> hashes = new LinkedHashMap<>();
		private final long fileSize;
		private final long hashTime;
		private String error;

		public HashResult(Path filePath, long fileSize, long hashTime) {
			this.filePath = filePath;
			this.fileSize = fileSize;
			this.hashTime = hashTime;
		}

		public HashResult(Path filePath, String error) {
			this.filePath = filePath;
			this.fileSize = 0;
			this.hashTime = 0;
			this.error = error;
		}

		public void addHash(HashAlgorithm algorithm, String hash) {
			hashes.put(algorithm, hash);
		}

		public Map<HashAlgorithm, String> getHashes() {
			return hashes;
		}

		public String getHash(HashAlgorithm algorithm) {
			return hashes.get(algorithm);
		}

		public Path getFilePath() {
			return filePath;
		}

		public long getFileSize() {
			return fileSize;
		}

		public long getHashTime() {
			return hashTime;
		}

		public boolean hasError() {
			return error != null;
		}

		public String getError() {
			return error;
		}

		@Override
		public String toString() {
			if (hasError()) {
				return String.format("%s: ERROR - %s", filePath.getFileName(), error);
			}

			StringBuilder sb = new StringBuilder();
			sb.append(String.format("%s (%,d bytes, %dms):\n", filePath.getFileName(), fileSize, hashTime));
			for (Map.Entry<HashAlgorithm, String> entry : hashes.entrySet()) {
				sb.append(String.format("  %-12s: %s\n", entry.getKey().name(), entry.getValue()));
			}
			return sb.toString();
		}
	}

	private final HashOptions options;

	public GGUFHasher() {
		this(new HashOptions());
	}

	public GGUFHasher(HashOptions options) {
		this.options = options;
	}

	/**
	 * Hash a single file
	 */
	public HashResult hashFile(Path filePath) {
		long startTime = System.currentTimeMillis();

		try {
			if (!Files.exists(filePath)) {
				return new HashResult(filePath, "File not found");
			}

			if (!Files.isRegularFile(filePath)) {
				return new HashResult(filePath, "Not a regular file");
			}

			long fileSize = Files.size(filePath);
			Map<HashAlgorithm, String> hashes = new LinkedHashMap<>();

			// Calculate different types of hashes
			for (HashAlgorithm algorithm : options.algorithms) {
				String hash = calculateHash(filePath, algorithm);
				hashes.put(algorithm, hash);
			}

			long hashTime = System.currentTimeMillis() - startTime;
			HashResult result = new HashResult(filePath, fileSize, hashTime);
			hashes.forEach(result::addHash);

			if (options.verbose) {
				LOGGER.info(String.format("Hashed %s in %d ms", filePath.getFileName(), hashTime));
			}

			return result;

		} catch (Exception e) {
			return new HashResult(filePath, "Error: " + e.getMessage());
		}
	}

	/**
	 * Hash multiple files
	 */
	public List<HashResult> hashFiles(List<Path> filePaths) {
		if (options.threads == 1) {
			// Single-threaded processing
			List<HashResult> results = new ArrayList<>();
			for (Path path : filePaths) {
				results.add(hashFile(path));
			}
			return results;
		} else {
			// Multi-threaded processing
			return hashFilesParallel(filePaths);
		}
	}

	private List<HashResult> hashFilesParallel(List<Path> filePaths) {
		ExecutorService executor = Executors.newFixedThreadPool(options.threads);
		List<Future<HashResult>> futures = new ArrayList<>();

		// Submit all hashing tasks
		for (Path path : filePaths) {
			futures.add(executor.submit(() -> hashFile(path)));
		}

		// Collect results in order
		List<HashResult> results = new ArrayList<>();
		for (Future<HashResult> future : futures) {
			try {
				results.add(future.get());
			} catch (InterruptedException | ExecutionException e) {
				LOGGER.warning("Failed to get hash result: " + e.getMessage());
			}
		}

		executor.shutdown();
		return results;
	}

	/**
	 * Hash directory (optionally recursive)
	 */
	public List<HashResult> hashDirectory(Path directoryPath) throws IOException {
		if (!Files.isDirectory(directoryPath)) {
			throw new IllegalArgumentException("Path is not a directory: " + directoryPath);
		}

		List<Path> filePaths = new ArrayList<>();

		if (options.recursive) {
			Files.walk(directoryPath)
				.filter(Files::isRegularFile)
				.filter(p -> p.toString().endsWith(".gguf"))
				.forEach(filePaths::add);
		} else {
			Files.list(directoryPath)
				.filter(Files::isRegularFile)
				.filter(p -> p.toString().endsWith(".gguf"))
				.forEach(filePaths::add);
		}

		return hashFiles(filePaths);
	}

	private String calculateHash(Path filePath, HashAlgorithm algorithm) throws IOException, NoSuchAlgorithmException {
		switch (algorithm) {
			case SHA256:
			case SHA1:
			case MD5:
				return calculateStandardHash(filePath, algorithm.getAlgorithmName());

			case GGUF_CONTENT:
				return calculateGGUFContentHash(filePath);

			case GGUF_METADATA:
				return calculateGGUFMetadataHash(filePath);

			default:
				throw new IllegalArgumentException("Unsupported algorithm: " + algorithm);
		}
	}

	private String calculateStandardHash(Path filePath, String algorithm) throws IOException, NoSuchAlgorithmException {
		MessageDigest digest = MessageDigest.getInstance(algorithm);

		try (InputStream is = Files.newInputStream(filePath);
		     BufferedInputStream bis = new BufferedInputStream(is, options.bufferSize)) {

			byte[] buffer = new byte[options.bufferSize];
			int bytesRead;
			while ((bytesRead = bis.read(buffer)) != -1) {
				digest.update(buffer, 0, bytesRead);
			}
		}

		return bytesToHex(digest.digest());
	}

	private String calculateGGUFContentHash(Path filePath) throws IOException {
		try (GGUFReader reader = new GGUFReader(filePath)) {
			MessageDigest digest = MessageDigest.getInstance("SHA-256");

			// Hash tensor data only (skip metadata)
			Map<String, TensorInfo> tensors = reader.getTensorInfos();
			List<String> sortedNames = new ArrayList<>(tensors.keySet());
			Collections.sort(sortedNames); // Ensure consistent ordering

			try (RandomAccessFile raf = new RandomAccessFile(filePath.toFile(), "r")) {
				for (String tensorName : sortedNames) {
					TensorInfo tensor = tensors.get(tensorName);

					// Hash tensor name first
					digest.update(tensorName.getBytes());

					// Hash tensor metadata
					digest.update(longToBytes(tensor.getOffset()));
					digest.update(longToBytes(tensor.getSize()));
					digest.update(tensor.getGgmlType().name().getBytes());

					// Hash shape
					long[] shape = tensor.getShape();
					for (long dim : shape) {
						digest.update(longToBytes(dim));
					}

					// Hash actual tensor data
					raf.seek(tensor.getOffset());
					byte[] buffer = new byte[Math.min(options.bufferSize, (int) tensor.getSize())];
					long remaining = tensor.getSize();

					while (remaining > 0) {
						int toRead = (int) Math.min(buffer.length, remaining);
						int bytesRead = raf.read(buffer, 0, toRead);
						if (bytesRead == -1) break;

						digest.update(buffer, 0, bytesRead);
						remaining -= bytesRead;
					}
				}
			}

			return bytesToHex(digest.digest());

		} catch (Exception e) {
			throw new IOException("Failed to calculate GGUF content hash", e);
		}
	}

	private String calculateGGUFMetadataHash(Path filePath) throws IOException {
		try (GGUFReader reader = new GGUFReader(filePath)) {
			MessageDigest digest = MessageDigest.getInstance("SHA-256");

			// Hash metadata only
			Map<String, GGUFValue> metadata = reader.getMetadata();
			List<String> sortedKeys = new ArrayList<>(metadata.keySet());
			Collections.sort(sortedKeys); // Ensure consistent ordering

			for (String key : sortedKeys) {
				GGUFValue value = metadata.get(key);

				// Hash key
				digest.update(key.getBytes());

				// Hash value type
				digest.update(value.getType().name().getBytes());

				// Hash value content
				hashGGUFValue(digest, value);
			}

			return bytesToHex(digest.digest());

		} catch (Exception e) {
			throw new IOException("Failed to calculate GGUF metadata hash", e);
		}
	}

	private void hashGGUFValue(MessageDigest digest, GGUFValue value) {
		Object val = value.getValue();
		if (val == null) {
			digest.update("null".getBytes());
			return;
		}

		switch (value.getType()) {
			case STRING:
				digest.update(((String) val).getBytes());
				break;

			case BOOL:
				digest.update(((Boolean) val) ? (byte) 1 : (byte) 0);
				break;

			case UINT8:
			case INT8:
				digest.update((byte) ((Number) val).intValue());
				break;

			case UINT16:
			case INT16:
				digest.update(shortToBytes((short) ((Number) val).intValue()));
				break;

			case UINT32:
			case INT32:
				digest.update(intToBytes(((Number) val).intValue()));
				break;

			case UINT64:
			case INT64:
				digest.update(longToBytes(((Number) val).longValue()));
				break;

			case FLOAT32:
				digest.update(intToBytes(Float.floatToIntBits(((Number) val).floatValue())));
				break;

			case FLOAT64:
				digest.update(longToBytes(Double.doubleToLongBits(((Number) val).doubleValue())));
				break;

			case ARRAY:
				Object[] array = (Object[]) val;
				digest.update(intToBytes(array.length));
				for (Object item : array) {
					if (item instanceof GGUFValue) {
						hashGGUFValue(digest, (GGUFValue) item);
					} else {
						digest.update(item.toString().getBytes());
					}
				}
				break;

			default:
				digest.update(val.toString().getBytes());
		}
	}

	private byte[] longToBytes(long value) {
		return new byte[] {
			(byte) (value >>> 56),
			(byte) (value >>> 48),
			(byte) (value >>> 40),
			(byte) (value >>> 32),
			(byte) (value >>> 24),
			(byte) (value >>> 16),
			(byte) (value >>> 8),
			(byte) value
		};
	}

	private byte[] intToBytes(int value) {
		return new byte[] {
			(byte) (value >>> 24),
			(byte) (value >>> 16),
			(byte) (value >>> 8),
			(byte) value
		};
	}

	private byte[] shortToBytes(short value) {
		return new byte[] {
			(byte) (value >>> 8),
			(byte) value
		};
	}

	private String bytesToHex(byte[] bytes) {
		StringBuilder result = new StringBuilder();
		for (byte b : bytes) {
			result.append(String.format("%02x", b));
		}
		return result.toString();
	}

	/**
	 * Save hash results to file
	 */
	public void saveResults(List<HashResult> results, String outputPath) throws IOException {
		try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
			for (HashResult result : results) {
				if (result.hasError()) {
					writer.printf("# ERROR: %s - %s%n", result.getFilePath(), result.getError());
				} else {
					for (Map.Entry<HashAlgorithm, String> entry : result.getHashes().entrySet()) {
						writer.printf("%s  %s%n", entry.getValue(), result.getFilePath());
					}
				}
			}
		}
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		if (args.length == 0) {
			printUsage();
			System.exit(1);
		}

		try {
			HashOptions options = new HashOptions();
			List<String> inputPaths = new ArrayList<>();

			// Parse arguments
			for (int i = 0; i < args.length; i++) {
				switch (args[i]) {
					case "--algorithm":
					case "-a":
						if (i + 1 < args.length) {
							String[] algos = args[++i].split(",");
							HashAlgorithm[] algorithms = new HashAlgorithm[algos.length];
							for (int j = 0; j < algos.length; j++) {
								algorithms[j] = HashAlgorithm.valueOf(algos[j].toUpperCase());
							}
							options.algorithms(algorithms);
						}
						break;
					case "--recursive":
					case "-r":
						options.recursive(true);
						break;
					case "--verbose":
					case "-v":
						options.verbose(true);
						break;
					case "--output":
					case "-o":
						if (i + 1 < args.length) {
							options.outputFile(args[++i]);
						}
						break;
					case "--threads":
					case "-t":
						if (i + 1 < args.length) {
							options.threads(Integer.parseInt(args[++i]));
						}
						break;
					case "--buffer-size":
						if (i + 1 < args.length) {
							options.bufferSize(Integer.parseInt(args[++i]) * 1024 * 1024);
						}
						break;
					case "--help":
					case "-h":
						printUsage();
						System.exit(0);
						break;
					default:
						if (!args[i].startsWith("-")) {
							inputPaths.add(args[i]);
						}
				}
			}

			if (inputPaths.isEmpty()) {
				System.err.println("Error: No input files specified");
				printUsage();
				System.exit(1);
			}

			GGUFHasher hasher = new GGUFHasher(options);
			List<HashResult> allResults = new ArrayList<>();

			for (String inputPath : inputPaths) {
				Path path = Paths.get(inputPath);
				if (Files.isDirectory(path)) {
					allResults.addAll(hasher.hashDirectory(path));
				} else {
					allResults.add(hasher.hashFile(path));
				}
			}

			// Print results
			for (HashResult result : allResults) {
				System.out.println(result);
			}

			// Save to file if requested
			if (options.outputFile) {
				hasher.saveResults(allResults, options.outputPath);
				System.out.println("Results saved to: " + options.outputPath);
			}

		} catch (Exception e) {
			System.err.println("Error: " + e.getMessage());
			if (Arrays.asList(args).contains("--verbose") || Arrays.asList(args).contains("-v")) {
				e.printStackTrace();
			}
			System.exit(1);
		}
	}

	private static void printUsage() {
		System.out.println("Usage: GGUFHasher [options] <file_or_directory>...");
		System.out.println();
		System.out.println("Calculate checksums for GGUF files.");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  -a, --algorithm <algos>  Hash algorithms (SHA256,SHA1,MD5,GGUF_CONTENT,GGUF_METADATA)");
		System.out.println("  -r, --recursive          Process directories recursively");
		System.out.println("  -v, --verbose            Verbose output");
		System.out.println("  -o, --output <file>      Save results to file");
		System.out.println("  -t, --threads <n>        Number of threads (default: CPU cores)");
		System.out.println("  --buffer-size <mb>       Buffer size in MB (default: 16)");
		System.out.println("  -h, --help               Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  GGUFHasher model.gguf");
		System.out.println("  GGUFHasher -a SHA256,GGUF_CONTENT *.gguf");
		System.out.println("  GGUFHasher -r -o checksums.txt models/");
		System.out.println("  GGUFHasher -v -t 8 large_model.gguf");
	}
}