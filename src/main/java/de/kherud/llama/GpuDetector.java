package de.kherud.llama;
import static java.lang.System.Logger.Level.DEBUG;

/**
 * Automatic GPU detection and configuration for optimal performance defaults.
 * Intelligently detects available GPU capabilities and suggests the best settings.
 */
public class GpuDetector {
	private static final System.Logger logger = System.getLogger(GpuDetector.class.getName());
	private static final int MAX_GPU_LAYERS = 999; // Let llama.cpp decide the actual limit

	private static GpuInfo cachedGpuInfo = null;

	public record GpuInfo(boolean cudaAvailable, long totalMemoryMB, String deviceName, int recommendedLayers, boolean shouldUseGpu) {

	@Override
		public String toString() {
			if (!cudaAvailable) {
				return "GPU: Not available (using CPU)";
			}
			return String.format("GPU: %s (%.1f GB VRAM) - Recommended layers: %d",
				deviceName, totalMemoryMB / 1024.0, recommendedLayers);
		}
	}

	/**
	 * Detect GPU capabilities and determine optimal configuration.
	 * Results are cached for subsequent calls.
	 */
	public static GpuInfo detectGpu() {
		if (cachedGpuInfo != null) {
			return cachedGpuInfo;
		}

		// Try to create a minimal test model to detect GPU capabilities
		// IMPORTANT: Use direct native call to avoid recursion
		try {
			// Create a temporary model to trigger GPU detection logs
			ModelParameters testParams = new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(1) // Try minimal GPU usage first
				.setCtxSize(64); // Minimal context for testing

			GpuDetectionLogger logger = new GpuDetectionLogger();
			LlamaModel.setLogger(null, logger::logMessage);

			// Use direct native loading to avoid recursion
			try (LlamaModel testModel = new LlamaModel()) {
				testModel.loadModelDirect(testParams.toArray()); // Direct call without auto-detection

				// Give logger time to capture messages
				Thread.sleep(50);

				cachedGpuInfo = logger.buildGpuInfo();

			} finally {
				// Reset logger
				LlamaModel.setLogger(null, null);
			}

		} catch (Exception e) {
			// Fallback: assume no GPU available
			logger.log(DEBUG, "GPU detection failed, using CPU-only mode: " + e.getMessage());
			cachedGpuInfo = new GpuInfo(false, 0, "None", 0, false);
		}

		return cachedGpuInfo;
	}

	/**
	 * Apply intelligent defaults to ModelParameters based on detected GPU capabilities.
	 */
	public static ModelParameters applyIntelligentDefaults(ModelParameters params) {
		GpuInfo gpu = detectGpu();

		logger.log(DEBUG, "🔍 " + gpu);

		if (gpu.shouldUseGpu && gpu.recommendedLayers > 0) {
			// Only set GPU layers if not explicitly configured by user
			if (!hasExplicitGpuLayers(params)) {
				params.setGpuLayers(gpu.recommendedLayers);
				logger.log(DEBUG, "🚀 Auto-configured GPU layers: " + gpu.recommendedLayers +
					" (for " + String.format("%.1f", gpu.totalMemoryMB / 1024.0) + " GB GPU)");
			}
		} else {
			logger.log(DEBUG, "💻 Using CPU-only mode");
		}

		// Apply other intelligent defaults
		params = applyOtherDefaults(params, gpu);

		return params;
	}

	/**
	 * Check if user has explicitly set GPU layers to avoid overriding their choice.
	 */
	private static boolean hasExplicitGpuLayers(ModelParameters params) {
		// Check if --gpu-layers parameter was explicitly set
		return params.parameters.containsKey("--gpu-layers");
	}

	/**
	 * Apply other intelligent defaults based on GPU capabilities.
	 */
	private static ModelParameters applyOtherDefaults(ModelParameters params, GpuInfo gpu) {
		// Flash attention for supported GPUs
		if (gpu.shouldUseGpu && gpu.totalMemoryMB > 8192) { // 8GB+ GPUs
			if (!params.parameters.containsKey("--flash-attn")) {
				params.enableFlashAttn();
				logger.log(DEBUG, "⚡ Enabled Flash Attention for large GPU");
			}
		}

		// Batch size optimization
		if (!params.parameters.containsKey("--batch-size")) {
			int batchSize = gpu.shouldUseGpu ? 512 : 256; // Larger batches for GPU
			params.setBatchSize(batchSize);
		}

		// Context size recommendation
		if (!params.parameters.containsKey("--ctx-size")) {
			int contextSize = gpu.totalMemoryMB > 16384 ? 4096 :  // 16GB+ GPU
			                 gpu.totalMemoryMB > 8192 ? 2048 :    // 8GB+ GPU
			                 gpu.shouldUseGpu ? 1024 : 512;       // Other GPU or CPU
			params.setCtxSize(contextSize);
			logger.log(DEBUG, "📝 Auto-configured context size: " + contextSize);
		}

		return params;
	}

	/**
	 * Logger to capture GPU detection information from llama.cpp logs.
	 */
	private static class GpuDetectionLogger {
		private boolean cudaFound = false;
		private String deviceName = "Unknown";
		private long vramMB = 0;
		private int detectedLayers = 0;

		void logMessage(LogLevel level, String message) {
			// Detect CUDA availability
			if (message.contains("found") && message.contains("CUDA devices")) {
				cudaFound = true;
			}

			// Extract GPU device name
			if (message.contains("Device 0:") && message.contains("NVIDIA")) {
				int start = message.indexOf("Device 0:") + 10;
				int end = message.indexOf(",", start);
				if (end > start) {
					deviceName = message.substring(start, end).trim();
				}
			}

			// Extract VRAM information
			if (message.contains("MiB free")) {
				try {
					String[] parts = message.split(" ");
					for (int i = 0; i < parts.length - 1; i++) {
						if (parts[i + 1].equals("MiB") && parts[i + 2].equals("free")) {
							vramMB = Long.parseLong(parts[i]);
							break;
						}
					}
				} catch (Exception e) {
					// Ignore parsing errors
				}
			}

			// Extract successful layer offloading
			if (message.contains("offloaded") && message.contains("layers to GPU")) {
				try {
					String[] parts = message.split(" ");
					for (int i = 0; i < parts.length - 1; i++) {
						if (parts[i].equals("offloaded")) {
							String layerInfo = parts[i + 1];
							if (layerInfo.contains("/")) {
								detectedLayers = Integer.parseInt(layerInfo.split("/")[0]);
							}
							break;
						}
					}
				} catch (Exception e) {
					// Ignore parsing errors
				}
			}
		}

		GpuInfo buildGpuInfo() {
			if (!cudaFound) {
				return new GpuInfo(false, 0, "None", 0, false);
			}

			// Determine recommended layers based on VRAM and detected capability
			int recommendedLayers = calculateRecommendedLayers(vramMB);
			boolean shouldUse = cudaFound && vramMB > 2048; // At least 2GB for meaningful GPU usage

			return new GpuInfo(cudaFound, vramMB, deviceName, recommendedLayers, shouldUse);
		}

		private int calculateRecommendedLayers(long vramMB) {
			if (vramMB >= 24000) return MAX_GPU_LAYERS; // 24GB+ - Use all layers
			if (vramMB >= 16000) return 48;             // 16GB+ - Most layers
			if (vramMB >= 12000) return 40;             // 12GB+ - Many layers
			if (vramMB >= 8000)  return 32;             // 8GB+  - Good coverage
			if (vramMB >= 6000)  return 24;             // 6GB+  - Moderate coverage
			if (vramMB >= 4000)  return 16;             // 4GB+  - Some layers
			if (vramMB >= 2000)  return 8;              // 2GB+  - Few layers
			return 0; // Less than 2GB - stick to CPU
		}
	}
}
