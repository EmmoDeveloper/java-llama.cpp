package de.kherud.llama;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

import static java.lang.System.Logger.Level.DEBUG;
/**
 * Utilities for managing threading configuration profiles.
 * Allows saving, loading, and managing threading configurations.
 */
public class ThreadingConfigUtils {
	private static final System.Logger logger = System.getLogger(ThreadingConfigUtils.class.getName());
	private static final String CONFIG_DIR = System.getProperty("user.home") + "/.java-llama-cpp";
	private static final String THREADING_CONFIG_FILE = CONFIG_DIR + "/threading.properties";

	/**
	 * Save a threading configuration profile.
	 *
	 * @param profileName Name of the profile
	 * @param config Threading configuration
	 */
	public static void saveThreadingProfile(String profileName, ThreadingConfig config) {
		try {
			createConfigDirectory();

			Properties props = loadThreadingProperties();
			String prefix = "profile." + profileName + ".";

			props.setProperty(prefix + "threads", String.valueOf(config.threads));
			props.setProperty(prefix + "batchThreads", String.valueOf(config.batchThreads));
			props.setProperty(prefix + "workloadType", config.workloadType.name());
			props.setProperty(prefix + "description", config.description);

			saveThreadingProperties(props);
			logger.log(DEBUG, "💾 Saved threading profile: " + profileName);

		} catch (IOException e) {
			System.err.println("Failed to save threading profile: " + e.getMessage());
		}
	}

	/**
	 * Load a threading configuration profile.
	 *
	 * @param profileName Name of the profile
	 * @return Threading configuration, or null if not found
	 */
	public static ThreadingConfig loadThreadingProfile(String profileName) {
		try {
			Properties props = loadThreadingProperties();
			String prefix = "profile." + profileName + ".";

			String threadsStr = props.getProperty(prefix + "threads");
			String batchThreadsStr = props.getProperty(prefix + "batchThreads");
			String workloadTypeStr = props.getProperty(prefix + "workloadType");
			String description = props.getProperty(prefix + "description", "");

			if (threadsStr == null || batchThreadsStr == null || workloadTypeStr == null) {
				return null;
			}

			int threads = Integer.parseInt(threadsStr);
			int batchThreads = Integer.parseInt(batchThreadsStr);
			ThreadingOptimizer.WorkloadType workloadType = ThreadingOptimizer.WorkloadType.valueOf(workloadTypeStr);

			return new ThreadingConfig(threads, batchThreads, workloadType, description);

		} catch (Exception e) {
			System.err.println("Failed to load threading profile '" + profileName + "': " + e.getMessage());
			return null;
		}
	}

	/**
	 * List all saved threading profiles.
	 *
	 * @return Array of profile names
	 */
	public static String[] listThreadingProfiles() {
		try {
			Properties props = loadThreadingProperties();
			return props.stringPropertyNames().stream()
					.filter(key -> key.startsWith("profile.") && key.endsWith(".threads"))
					.map(key -> key.substring(8, key.lastIndexOf(".threads")))
					.sorted()
					.toArray(String[]::new);
		} catch (IOException e) {
			System.err.println("Failed to list threading profiles: " + e.getMessage());
			return new String[0];
		}
	}

	/**
	 * Delete a threading profile.
	 *
	 * @param profileName Name of the profile to delete
	 * @return true if deleted successfully
	 */
	public static boolean deleteThreadingProfile(String profileName) {
		try {
			Properties props = loadThreadingProperties();
			String prefix = "profile." + profileName + ".";

			boolean found = false;
			for (String key : props.stringPropertyNames().toArray(new String[0])) {
				if (key.startsWith(prefix)) {
					props.remove(key);
					found = true;
				}
			}

			if (found) {
				saveThreadingProperties(props);
				logger.log(DEBUG, "🗑️ Deleted threading profile: " + profileName);
				return true;
			}

			return false;
		} catch (IOException e) {
			System.err.println("Failed to delete threading profile: " + e.getMessage());
			return false;
		}
	}

	/**
	 * Apply a threading profile to model parameters.
	 *
	 * @param params Model parameters to modify
	 * @param profileName Profile name to apply
	 * @return Modified parameters, or original if profile not found
	 */
	public static ModelParameters applyThreadingProfile(ModelParameters params, String profileName) {
		ThreadingConfig config = loadThreadingProfile(profileName);
		if (config == null) {
			System.err.println("Threading profile '" + profileName + "' not found");
			return params;
		}

		params.setThreads(config.threads);
		params.setThreadsBatch(config.batchThreads);

		logger.log(DEBUG, "🔧 Applied threading profile: " + profileName +
						  " (threads: " + config.threads + ", batch: " + config.batchThreads + ")");

		return params;
	}

	/**
	 * Create default threading profiles.
	 */
	public static void createDefaultProfiles() {
		// High-performance profile
		saveThreadingProfile("high-performance", new ThreadingConfig(
			Runtime.getRuntime().availableProcessors(),
			Runtime.getRuntime().availableProcessors() + 2,
			ThreadingOptimizer.WorkloadType.GENERAL,
			"Maximum performance using all available CPU cores"
		));

		// Balanced profile
		saveThreadingProfile("balanced", new ThreadingConfig(
			Math.max(Runtime.getRuntime().availableProcessors() / 2, 2),
			Math.max(Runtime.getRuntime().availableProcessors() / 2 + 1, 3),
			ThreadingOptimizer.WorkloadType.GENERAL,
			"Balanced performance leaving CPU capacity for other tasks"
		));

		// Low-resource profile
		saveThreadingProfile("low-resource", new ThreadingConfig(
			2, 2,
			ThreadingOptimizer.WorkloadType.GENERAL,
			"Minimal CPU usage for resource-constrained environments"
		));

		// Embedding-optimized profile
		int embeddingThreads = Math.max((Runtime.getRuntime().availableProcessors() * 3) / 4, 2);
		saveThreadingProfile("embedding-optimized", new ThreadingConfig(
			embeddingThreads,
			embeddingThreads,
			ThreadingOptimizer.WorkloadType.EMBEDDING,
			"Optimized for high-throughput embedding generation"
		));

		logger.log(DEBUG, "✅ Created default threading profiles");
	}

	/**
	 * Print threading profile information.
	 *
	 * @param profileName Profile name to display
	 */
	public static void printProfile(String profileName) {
		ThreadingConfig config = loadThreadingProfile(profileName);
		if (config == null) {
			logger.log(DEBUG, "Profile '" + profileName + "' not found");
			return;
		}

		logger.log(DEBUG, "📋 Threading Profile: " + profileName);
		logger.log(DEBUG, "   Threads: " + config.threads);
		logger.log(DEBUG, "   Batch Threads: " + config.batchThreads);
		logger.log(DEBUG, "   Workload Type: " + config.workloadType);
		logger.log(DEBUG, "   Description: " + config.description);
	}

	/**
	 * Print all threading profiles.
	 */
	public static void printAllProfiles() {
		String[] profiles = listThreadingProfiles();
		if (profiles.length == 0) {
			logger.log(DEBUG, "No threading profiles found. Use createDefaultProfiles() to create some.");
			return;
		}

		logger.log(DEBUG, "\n📋 Available Threading Profiles:");
		for (String profile : profiles) {
			ThreadingConfig config = loadThreadingProfile(profile);
			if (config != null) {
				logger.log(DEBUG, "   • %s: %d threads, %d batch (%s)",
					profile, config.threads, config.batchThreads, config.workloadType);
			}
		}
	}

	private static void createConfigDirectory() {
		File dir = new File(CONFIG_DIR);
		if (!dir.exists()) {
			dir.mkdirs();
		}
	}

	private static Properties loadThreadingProperties() throws IOException {
		Properties props = new Properties();
		File file = new File(THREADING_CONFIG_FILE);
		if (file.exists()) {
			try (FileInputStream fis = new FileInputStream(file)) {
				props.load(fis);
			}
		}
		return props;
	}

	private static void saveThreadingProperties(Properties props) throws IOException {
		createConfigDirectory();
		try (FileOutputStream fos = new FileOutputStream(THREADING_CONFIG_FILE)) {
			props.store(fos, "Java-llama.cpp Threading Configuration");
		}
	}

	/**
	 * Threading configuration data class.
	 */
	public static class ThreadingConfig {
		public final int threads;
		public final int batchThreads;
		public final ThreadingOptimizer.WorkloadType workloadType;
		public final String description;

		public ThreadingConfig(int threads, int batchThreads, ThreadingOptimizer.WorkloadType workloadType, String description) {
			this.threads = threads;
			this.batchThreads = batchThreads;
			this.workloadType = workloadType;
			this.description = description;
		}
	}
}
