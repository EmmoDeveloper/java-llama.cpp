package de.kherud.llama.huggingface;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.regex.Pattern;

/**
 * HuggingFace model downloader and manager.
 *
 * Equivalent to huggingface_hub Python library functionality - downloads models,
 * tokenizers, and other artifacts from HuggingFace Hub with resume support,
 * authentication, and metadata handling.
 */
public class HuggingFaceDownloader {
	private static final Logger LOGGER = Logger.getLogger(HuggingFaceDownloader.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();
	private static final String HF_HUB_URL = "https://huggingface.co";
	private static final String HF_API_URL = "https://api.huggingface.co";

	public static class DownloadConfig {
		private String token = null; // HF token for private models
		private boolean resumeDownload = true;
		private int maxConcurrentDownloads = 4;
		private int timeout = 300000; // 5 minutes
		private boolean verbose = false;
		private String userAgent = "java-llama.cpp/1.0";
		private Set<String> allowedFileExtensions = new HashSet<>(Arrays.asList(
			".bin", ".safetensors", ".json", ".txt", ".py", ".md", ".gitattributes"
		));
		private Pattern filenameFilter = null;
		private boolean includeTokenizer = true;
		private boolean includeConfig = true;
		private boolean includeReadme = false;

		public DownloadConfig token(String token) {
			this.token = token;
			return this;
		}

		public DownloadConfig resumeDownload(boolean resume) {
			this.resumeDownload = resume;
			return this;
		}

		public DownloadConfig maxConcurrentDownloads(int max) {
			this.maxConcurrentDownloads = max;
			return this;
		}

		public DownloadConfig timeout(int timeout) {
			this.timeout = timeout;
			return this;
		}

		public DownloadConfig verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public DownloadConfig userAgent(String userAgent) {
			this.userAgent = userAgent;
			return this;
		}

		public DownloadConfig allowedFileExtensions(Set<String> extensions) {
			this.allowedFileExtensions = new HashSet<>(extensions);
			return this;
		}

		public DownloadConfig filenameFilter(Pattern filter) {
			this.filenameFilter = filter;
			return this;
		}

		public DownloadConfig includeTokenizer(boolean include) {
			this.includeTokenizer = include;
			return this;
		}

		public DownloadConfig includeConfig(boolean include) {
			this.includeConfig = include;
			return this;
		}

		public DownloadConfig includeReadme(boolean include) {
			this.includeReadme = include;
			return this;
		}
	}

	public static class ModelInfo {
		public String id;
		public String author;
		public String sha;
		public long lastModified;
		public List<String> tags = new ArrayList<>();
		public Map<String, Object> config = new HashMap<>();
		public List<FileInfo> files = new ArrayList<>();
		public boolean isPrivate;
		public boolean isGated;

		public static class FileInfo {
			public String filename;
			public long size;
			public String sha;
			public String downloadUrl;
			public boolean isLfs;
		}
	}

	public static class DownloadResult {
		public String modelId;
		public Path downloadPath;
		public List<Path> downloadedFiles = new ArrayList<>();
		public long totalSize;
		public long downloadTime;
		public boolean success;
		public String error;
		public ModelInfo modelInfo;
	}

	private final DownloadConfig config;
	private final ExecutorService executor;
	private final String cacheDir;

	public HuggingFaceDownloader() {
		this(new DownloadConfig());
	}

	public HuggingFaceDownloader(DownloadConfig config) {
		this.config = config;
		this.executor = Executors.newFixedThreadPool(config.maxConcurrentDownloads);
		this.cacheDir = System.getProperty("user.home") + "/.cache/huggingface/hub";
	}

	/**
	 * Download a model from HuggingFace Hub
	 */
	public DownloadResult downloadModel(String modelId, Path outputDir) throws IOException {
		LOGGER.info("Starting download for model: " + modelId);
		DownloadResult result = new DownloadResult();
		result.modelId = modelId;
		result.downloadPath = outputDir;

		long startTime = System.currentTimeMillis();

		try {
			// Create output directory
			Files.createDirectories(outputDir);

			// Get model information
			result.modelInfo = getModelInfo(modelId);

			// Filter files to download
			List<ModelInfo.FileInfo> filesToDownload = filterFiles(result.modelInfo.files);

			if (filesToDownload.isEmpty()) {
				throw new IOException("No files to download after filtering");
			}

			LOGGER.info("Found " + filesToDownload.size() + " files to download");

			// Download files
			downloadFiles(filesToDownload, outputDir, result);

			result.downloadTime = System.currentTimeMillis() - startTime;
			result.success = true;

			LOGGER.info("Download completed successfully in " + result.downloadTime + "ms");
			LOGGER.info("Downloaded " + result.downloadedFiles.size() + " files");

		} catch (Exception e) {
			result.success = false;
			result.error = e.getMessage();
			LOGGER.log(Level.SEVERE, "Download failed for model: " + modelId, e);
		}

		return result;
	}

	/**
	 * Get model information from HuggingFace API
	 */
	public ModelInfo getModelInfo(String modelId) throws IOException {
		String apiUrl = HF_API_URL + "/api/models/" + modelId;
		HttpURLConnection conn = createConnection(apiUrl, "GET");

		int responseCode = conn.getResponseCode();
		if (responseCode != 200) {
			throw new IOException("Failed to get model info: HTTP " + responseCode);
		}

		String response = readResponse(conn);
		JsonNode json = MAPPER.readTree(response);

		ModelInfo info = new ModelInfo();
		info.id = json.get("id").asText();
		info.author = json.path("author").asText();
		info.sha = json.path("sha").asText();
		info.lastModified = json.path("lastModified").asLong();
		info.isPrivate = json.path("private").asBoolean();
		info.isGated = json.path("gated").asBoolean();

		// Parse tags
		if (json.has("tags")) {
			for (JsonNode tag : json.get("tags")) {
				info.tags.add(tag.asText());
			}
		}

		// Get file listing
		getFileList(modelId, info);

		return info;
	}

	private void getFileList(String modelId, ModelInfo info) throws IOException {
		String apiUrl = HF_API_URL + "/api/models/" + modelId + "/tree/main";
		HttpURLConnection conn = createConnection(apiUrl, "GET");

		int responseCode = conn.getResponseCode();
		if (responseCode != 200) {
			throw new IOException("Failed to get file list: HTTP " + responseCode);
		}

		String response = readResponse(conn);
		JsonNode json = MAPPER.readTree(response);

		for (JsonNode fileNode : json) {
			if (fileNode.get("type").asText().equals("file")) {
				ModelInfo.FileInfo fileInfo = new ModelInfo.FileInfo();
				fileInfo.filename = fileNode.get("path").asText();
				fileInfo.size = fileNode.path("size").asLong();
				fileInfo.sha = fileNode.path("oid").asText();
				fileInfo.isLfs = fileNode.path("lfs").asBoolean();

				// Construct download URL
				if (fileInfo.isLfs) {
					fileInfo.downloadUrl = HF_HUB_URL + "/" + modelId + "/resolve/main/" + fileInfo.filename;
				} else {
					fileInfo.downloadUrl = HF_HUB_URL + "/" + modelId + "/raw/main/" + fileInfo.filename;
				}

				info.files.add(fileInfo);
			}
		}
	}

	private List<ModelInfo.FileInfo> filterFiles(List<ModelInfo.FileInfo> files) {
		List<ModelInfo.FileInfo> filtered = new ArrayList<>();

		for (ModelInfo.FileInfo file : files) {
			// Apply filename filter
			if (config.filenameFilter != null && !config.filenameFilter.matcher(file.filename).matches()) {
				continue;
			}

			// Check file extension
			String extension = getFileExtension(file.filename);
			if (!config.allowedFileExtensions.contains(extension)) {
				continue;
			}

			// Apply specific filters
			if (!config.includeTokenizer && isTokenizerFile(file.filename)) {
				continue;
			}

			if (!config.includeConfig && isConfigFile(file.filename)) {
				continue;
			}

			if (!config.includeReadme && isReadmeFile(file.filename)) {
				continue;
			}

			filtered.add(file);
		}

		return filtered;
	}

	private boolean isTokenizerFile(String filename) {
		String lower = filename.toLowerCase();
		return lower.contains("tokenizer") || lower.equals("vocab.txt") ||
			   lower.equals("merges.txt") || lower.equals("special_tokens_map.json");
	}

	private boolean isConfigFile(String filename) {
		String lower = filename.toLowerCase();
		return lower.equals("config.json") || lower.equals("generation_config.json") ||
			   lower.equals("model_config.json");
	}

	private boolean isReadmeFile(String filename) {
		String lower = filename.toLowerCase();
		return lower.equals("readme.md") || lower.equals("model_card.md");
	}

	private String getFileExtension(String filename) {
		int lastDot = filename.lastIndexOf('.');
		if (lastDot > 0) {
			return filename.substring(lastDot);
		}
		return "";
	}

	private void downloadFiles(List<ModelInfo.FileInfo> files, Path outputDir, DownloadResult result) throws InterruptedException {
		List<Future<Path>> downloadTasks = new ArrayList<>();
		result.totalSize = files.stream().mapToLong(f -> f.size).sum();

		for (ModelInfo.FileInfo file : files) {
			Future<Path> task = executor.submit(() -> downloadFile(file, outputDir));
			downloadTasks.add(task);
		}

		// Wait for all downloads to complete
		for (Future<Path> task : downloadTasks) {
			try {
				Path downloadedFile = task.get();
				if (downloadedFile != null) {
					result.downloadedFiles.add(downloadedFile);
				}
			} catch (ExecutionException e) {
				LOGGER.log(Level.WARNING, "File download failed", e.getCause());
			}
		}
	}

	private Path downloadFile(ModelInfo.FileInfo file, Path outputDir) throws IOException {
		Path outputFile = outputDir.resolve(file.filename);
		Files.createDirectories(outputFile.getParent());

		if (config.verbose) {
			LOGGER.info("Downloading: " + file.filename + " (" + formatSize(file.size) + ")");
		}

		// Check if file already exists and resume is enabled
		long existingSize = 0;
		if (config.resumeDownload && Files.exists(outputFile)) {
			existingSize = Files.size(outputFile);
			if (existingSize == file.size) {
				if (config.verbose) {
					LOGGER.info("File already exists and is complete: " + file.filename);
				}
				return outputFile;
			}
		}

		HttpURLConnection conn = createConnection(file.downloadUrl, "GET");

		// Set range header for resume
		if (existingSize > 0) {
			conn.setRequestProperty("Range", "bytes=" + existingSize + "-");
			if (config.verbose) {
				LOGGER.info("Resuming download from byte " + existingSize);
			}
		}

		int responseCode = conn.getResponseCode();
		if (responseCode != 200 && responseCode != 206) {
			throw new IOException("Download failed for " + file.filename + ": HTTP " + responseCode);
		}

		// Download file
		try (InputStream in = conn.getInputStream();
			 OutputStream out = Files.newOutputStream(outputFile,
				 existingSize > 0 ? StandardOpenOption.APPEND : StandardOpenOption.CREATE)) {

			byte[] buffer = new byte[8192];
			int bytesRead;
			long totalBytes = existingSize;

			while ((bytesRead = in.read(buffer)) != -1) {
				out.write(buffer, 0, bytesRead);
				totalBytes += bytesRead;

				if (config.verbose && totalBytes % (1024 * 1024) == 0) {
					double progress = (double) totalBytes / file.size * 100;
					LOGGER.info(String.format("Progress for %s: %.1f%%", file.filename, progress));
				}
			}
		}

		if (config.verbose) {
			LOGGER.info("Completed: " + file.filename);
		}

		return outputFile;
	}

	/**
	 * Search for models on HuggingFace Hub
	 */
	public List<ModelInfo> searchModels(String query, int limit) throws IOException {
		String apiUrl = HF_API_URL + "/api/models?search=" + query + "&limit=" + limit;
		HttpURLConnection conn = createConnection(apiUrl, "GET");

		int responseCode = conn.getResponseCode();
		if (responseCode != 200) {
			throw new IOException("Search failed: HTTP " + responseCode);
		}

		String response = readResponse(conn);
		JsonNode json = MAPPER.readTree(response);

		List<ModelInfo> models = new ArrayList<>();
		for (JsonNode modelNode : json) {
			ModelInfo info = new ModelInfo();
			info.id = modelNode.get("id").asText();
			info.author = modelNode.path("author").asText();
			info.lastModified = modelNode.path("lastModified").asLong();
			info.isPrivate = modelNode.path("private").asBoolean();

			if (modelNode.has("tags")) {
				for (JsonNode tag : modelNode.get("tags")) {
					info.tags.add(tag.asText());
				}
			}

			models.add(info);
		}

		return models;
	}

	/**
	 * Check if a model exists and is accessible
	 */
	public boolean modelExists(String modelId) {
		try {
			getModelInfo(modelId);
			return true;
		} catch (IOException e) {
			return false;
		}
	}

	/**
	 * Get cached model path if it exists
	 */
	public Optional<Path> getCachedModelPath(String modelId) {
		Path cacheModelDir = Paths.get(cacheDir, modelId.replace("/", "--"));
		if (Files.exists(cacheModelDir) && Files.isDirectory(cacheModelDir)) {
			return Optional.of(cacheModelDir);
		}
		return Optional.empty();
	}

	private HttpURLConnection createConnection(String url, String method) throws IOException {
		HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
		conn.setRequestMethod(method);
		conn.setConnectTimeout(config.timeout);
		conn.setReadTimeout(config.timeout);
		conn.setRequestProperty("User-Agent", config.userAgent);

		if (config.token != null) {
			conn.setRequestProperty("Authorization", "Bearer " + config.token);
		}

		return conn;
	}

	private String readResponse(HttpURLConnection conn) throws IOException {
		try (InputStream is = conn.getInputStream();
			 BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
			StringBuilder response = new StringBuilder();
			String line;
			while ((line = reader.readLine()) != null) {
				response.append(line);
			}
			return response.toString();
		}
	}

	private String formatSize(long bytes) {
		if (bytes < 1024) return bytes + " B";
		int exp = (int) (Math.log(bytes) / Math.log(1024));
		String pre = "KMGTPE".charAt(exp - 1) + "";
		return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
	}

	public void close() {
		if (executor != null && !executor.isShutdown()) {
			executor.shutdown();
			try {
				if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
					executor.shutdownNow();
				}
			} catch (InterruptedException e) {
				executor.shutdownNow();
				Thread.currentThread().interrupt();
			}
		}
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		if (args.length < 2) {
			printUsage();
			System.exit(1);
		}

		try {
			String command = args[0];
			DownloadConfig config = new DownloadConfig();

			// Parse options
			int argIndex = 1;
			while (argIndex < args.length && args[argIndex].startsWith("--")) {
				switch (args[argIndex]) {
					case "--token":
						if (argIndex + 1 < args.length) {
							config.token(args[++argIndex]);
						}
						break;
					case "--output":
					case "-o":
						// Handled per command
						break;
					case "--verbose":
					case "-v":
						config.verbose(true);
						break;
					case "--no-resume":
						config.resumeDownload(false);
						break;
					case "--concurrent":
						if (argIndex + 1 < args.length) {
							config.maxConcurrentDownloads(Integer.parseInt(args[++argIndex]));
						}
						break;
					case "--timeout":
						if (argIndex + 1 < args.length) {
							config.timeout(Integer.parseInt(args[++argIndex]));
						}
						break;
					case "--filter":
						if (argIndex + 1 < args.length) {
							config.filenameFilter(Pattern.compile(args[++argIndex]));
						}
						break;
					case "--no-tokenizer":
						config.includeTokenizer(false);
						break;
					case "--no-config":
						config.includeConfig(false);
						break;
					case "--include-readme":
						config.includeReadme(true);
						break;
					case "--help":
					case "-h":
						printUsage();
						System.exit(0);
						break;
				}
				argIndex++;
			}

			HuggingFaceDownloader downloader = new HuggingFaceDownloader(config);

			try {
				switch (command) {
					case "download":
						handleDownloadCommand(args, argIndex, downloader, config);
						break;
					case "info":
						handleInfoCommand(args, argIndex, downloader);
						break;
					case "search":
						handleSearchCommand(args, argIndex, downloader);
						break;
					case "list-cache":
						handleListCacheCommand(downloader);
						break;
					default:
						System.err.println("Unknown command: " + command);
						printUsage();
						System.exit(1);
				}
			} finally {
				downloader.close();
			}

		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "Command failed", e);
			System.exit(1);
		}
	}

	private static void handleDownloadCommand(String[] args, int startIndex,
											 HuggingFaceDownloader downloader, DownloadConfig config) throws IOException {
		if (startIndex >= args.length) {
			System.err.println("Model ID required for download command");
			System.exit(1);
		}

		String modelId = args[startIndex];
		Path outputDir = Paths.get("./models/" + modelId.replace("/", "_"));

		// Check for output directory override
		for (int i = 0; i < args.length - 1; i++) {
			if ("--output".equals(args[i]) || "-o".equals(args[i])) {
				outputDir = Paths.get(args[i + 1]);
				break;
			}
		}

		DownloadResult result = downloader.downloadModel(modelId, outputDir);

		if (result.success) {
			System.out.println("Download successful!");
			System.out.println("Model: " + result.modelId);
			System.out.println("Path: " + result.downloadPath);
			System.out.println("Files: " + result.downloadedFiles.size());
			System.out.println("Size: " + downloader.formatSize(result.totalSize));
			System.out.println("Time: " + result.downloadTime + "ms");
		} else {
			System.err.println("Download failed: " + result.error);
			System.exit(1);
		}
	}

	private static void handleInfoCommand(String[] args, int startIndex, HuggingFaceDownloader downloader) throws IOException {
		if (startIndex >= args.length) {
			System.err.println("Model ID required for info command");
			System.exit(1);
		}

		String modelId = args[startIndex];
		ModelInfo info = downloader.getModelInfo(modelId);

		System.out.println("=== MODEL INFORMATION ===");
		System.out.println("ID: " + info.id);
		System.out.println("Author: " + info.author);
		System.out.println("SHA: " + info.sha);
		System.out.println("Last Modified: " + new Date(info.lastModified));
		System.out.println("Private: " + info.isPrivate);
		System.out.println("Gated: " + info.isGated);
		System.out.println("Tags: " + String.join(", ", info.tags));
		System.out.println("Files: " + info.files.size());

		long totalSize = info.files.stream().mapToLong(f -> f.size).sum();
		System.out.println("Total Size: " + downloader.formatSize(totalSize));
	}

	private static void handleSearchCommand(String[] args, int startIndex, HuggingFaceDownloader downloader) throws IOException {
		if (startIndex >= args.length) {
			System.err.println("Search query required for search command");
			System.exit(1);
		}

		String query = args[startIndex];
		int limit = 10;

		// Check for limit override
		for (int i = 0; i < args.length - 1; i++) {
			if ("--limit".equals(args[i])) {
				limit = Integer.parseInt(args[i + 1]);
				break;
			}
		}

		List<ModelInfo> models = downloader.searchModels(query, limit);

		System.out.println("=== SEARCH RESULTS ===");
		System.out.println("Query: " + query);
		System.out.println("Results: " + models.size());
		System.out.println();

		for (ModelInfo model : models) {
			System.out.printf("%-40s %s%n", model.id, model.author);
			if (!model.tags.isEmpty()) {
				System.out.println("  Tags: " + String.join(", ", model.tags));
			}
		}
	}

	private static void handleListCacheCommand(HuggingFaceDownloader downloader) throws IOException {
		Path cacheDir = Paths.get(downloader.cacheDir);
		if (!Files.exists(cacheDir)) {
			System.out.println("No cache directory found");
			return;
		}

		System.out.println("=== CACHED MODELS ===");
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(cacheDir)) {
			for (Path entry : stream) {
				if (Files.isDirectory(entry)) {
					String modelId = entry.getFileName().toString().replace("--", "/");
					long size = Files.walk(entry)
						.filter(Files::isRegularFile)
						.mapToLong(p -> {
							try { return Files.size(p); }
							catch (IOException e) { return 0; }
						})
						.sum();

					System.out.printf("%-40s %s%n", modelId, downloader.formatSize(size));
				}
			}
		}
	}

	private static void printUsage() {
		System.out.println("Usage: HuggingFaceDownloader <command> [options] [args]");
		System.out.println();
		System.out.println("Download and manage models from HuggingFace Hub.");
		System.out.println();
		System.out.println("Commands:");
		System.out.println("  download <model_id>    Download a model");
		System.out.println("  info <model_id>        Show model information");
		System.out.println("  search <query>         Search for models");
		System.out.println("  list-cache             List cached models");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --token <token>        HuggingFace API token");
		System.out.println("  --output, -o <dir>     Output directory");
		System.out.println("  --verbose, -v          Verbose output");
		System.out.println("  --no-resume            Disable resume downloads");
		System.out.println("  --concurrent <n>       Max concurrent downloads (default: 4)");
		System.out.println("  --timeout <ms>         Request timeout (default: 300000)");
		System.out.println("  --filter <regex>       Filename filter regex");
		System.out.println("  --no-tokenizer         Skip tokenizer files");
		System.out.println("  --no-config            Skip configuration files");
		System.out.println("  --include-readme       Include README files");
		System.out.println("  --help, -h             Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  HuggingFaceDownloader download microsoft/DialoGPT-medium");
		System.out.println("  HuggingFaceDownloader info microsoft/DialoGPT-medium");
		System.out.println("  HuggingFaceDownloader search \"llama\"");
		System.out.println("  HuggingFaceDownloader download --token abc123 private/model");
	}
}