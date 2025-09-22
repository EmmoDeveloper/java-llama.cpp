package de.kherud.llama.gguf;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.*;
import java.util.logging.Logger;

/**
 * GGUF file inspection utility.
 *
 * Equivalent to gguf_dump.py - provides detailed information about GGUF file contents
 * including metadata, tensors, and file structure.
 */
public class GGUFInspector {
	private static final Logger LOGGER = Logger.getLogger(GGUFInspector.class.getName());

	public static class InspectionOptions {
		private boolean showMetadata = true;
		private boolean showTensors = true;
		private boolean showTensorData = false;
		private boolean showFileStructure = true;
		private boolean verbose = false;
		private boolean jsonOutput = false;
		private String filterKey = null;
		private int maxStringLength = 60;

		public InspectionOptions metadata(boolean show) {
			this.showMetadata = show;
			return this;
		}

		public InspectionOptions tensors(boolean show) {
			this.showTensors = show;
			return this;
		}

		public InspectionOptions tensorData(boolean show) {
			this.showTensorData = show;
			return this;
		}

		public InspectionOptions fileStructure(boolean show) {
			this.showFileStructure = show;
			return this;
		}

		public InspectionOptions verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public InspectionOptions jsonOutput(boolean json) {
			this.jsonOutput = json;
			return this;
		}

		public InspectionOptions filterKey(String key) {
			this.filterKey = key;
			return this;
		}

		public InspectionOptions maxStringLength(int length) {
			this.maxStringLength = length;
			return this;
		}
	}

	public static class InspectionResult {
		public final Map<String, Object> metadata = new LinkedHashMap<>();
		public final Map<String, TensorInfo> tensors = new LinkedHashMap<>();
		public final FileInfo fileInfo = new FileInfo();

		public static class FileInfo {
			public String endianness;
			public String hostEndianness;
			public long fileSize;
			public int version;
			public int tensorCount;
			public int metadataCount;
			public String checksum;
		}
	}

	private final GGUFReader reader;
	private final Path filePath;

	public GGUFInspector(Path filePath) throws IOException {
		this.filePath = filePath;
		this.reader = new GGUFReader(filePath);
	}

	public InspectionResult inspect() throws IOException {
		return inspect(new InspectionOptions());
	}

	public InspectionResult inspect(InspectionOptions options) throws IOException {
		InspectionResult result = new InspectionResult();

		// File structure information
		populateFileInfo(result.fileInfo);

		// Metadata
		if (options.showMetadata) {
			populateMetadata(result.metadata, options);
		}

		// Tensors
		if (options.showTensors) {
			populateTensors(result.tensors, options);
		}

		return result;
	}

	public void printInspection() throws IOException {
		printInspection(new InspectionOptions());
	}

	public void printInspection(InspectionOptions options) throws IOException {
		InspectionResult result = inspect(options);

		if (options.jsonOutput) {
			printJsonOutput(result);
		} else {
			printFormattedOutput(result, options);
		}
	}

	private void populateFileInfo(InspectionResult.FileInfo fileInfo) throws IOException {
		fileInfo.endianness = reader.getByteOrder() == ByteOrder.LITTLE_ENDIAN ? "LITTLE" : "BIG";
		fileInfo.hostEndianness = ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? "LITTLE" : "BIG";
		fileInfo.fileSize = filePath.toFile().length();
		fileInfo.version = reader.getVersion();
		fileInfo.tensorCount = reader.getTensorInfos().size();
		fileInfo.metadataCount = reader.getMetadata().size();
		fileInfo.checksum = calculateChecksum();
	}

	private void populateMetadata(Map<String, Object> metadata, InspectionOptions options) {
		Map<String, GGUFValue> rawMetadata = reader.getMetadata();

		for (Map.Entry<String, GGUFValue> entry : rawMetadata.entrySet()) {
			String key = entry.getKey();
			GGUFValue value = entry.getValue();

			// Apply filter if specified
			if (options.filterKey != null && !key.contains(options.filterKey)) {
				continue;
			}

			Object processedValue = processMetadataValue(value, options);
			metadata.put(key, processedValue);
		}
	}

	private Object processMetadataValue(GGUFValue value, InspectionOptions options) {
		switch (value.getType()) {
			case UINT8:
			case INT8:
			case UINT16:
			case INT16:
			case UINT32:
			case INT32:
			case UINT64:
			case INT64:
				return value.getValue();

			case FLOAT32:
			case FLOAT64:
				return value.getValue();

			case BOOL:
				return value.getValue();

			case STRING:
				String str = (String) value.getValue();
				if (str.length() > options.maxStringLength && !options.verbose) {
					return str.substring(0, options.maxStringLength) + "...";
				}
				return str;

			case ARRAY:
				Object[] array = (Object[]) value.getValue();
				List<Object> processedArray = new ArrayList<>();
				for (Object item : array) {
					if (item instanceof GGUFValue) {
						processedArray.add(processMetadataValue((GGUFValue) item, options));
					} else {
						processedArray.add(item);
					}
				}
				return processedArray;

			default:
				return value.getValue() != null ? value.getValue().toString() : "null";
		}
	}

	private void populateTensors(Map<String, TensorInfo> tensors, InspectionOptions options) {
		Map<String, TensorInfo> rawTensors = reader.getTensorInfos();

		for (Map.Entry<String, TensorInfo> entry : rawTensors.entrySet()) {
			String name = entry.getKey();
			TensorInfo tensorInfo = entry.getValue();

			// Apply filter if specified
			if (options.filterKey != null && !name.contains(options.filterKey)) {
				continue;
			}

			tensors.put(name, tensorInfo);
		}
	}

	private String calculateChecksum() {
		try {
			MessageDigest md = MessageDigest.getInstance("SHA-256");
			try (InputStream is = new FileInputStream(filePath.toFile())) {
				byte[] buffer = new byte[8192];
				int bytesRead;
				while ((bytesRead = is.read(buffer)) != -1) {
					md.update(buffer, 0, bytesRead);
				}
			}
			byte[] hash = md.digest();
			StringBuilder hexString = new StringBuilder();
			for (byte b : hash) {
				String hex = Integer.toHexString(0xff & b);
				if (hex.length() == 1) {
					hexString.append('0');
				}
				hexString.append(hex);
			}
			return hexString.toString();
		} catch (Exception e) {
			return "Error calculating checksum: " + e.getMessage();
		}
	}

	private void printJsonOutput(InspectionResult result) throws IOException {
		ObjectMapper mapper = new ObjectMapper();
		Map<String, Object> output = new LinkedHashMap<>();
		output.put("file_info", result.fileInfo);
		output.put("metadata", result.metadata);
		output.put("tensors", convertTensorsForJson(result.tensors));

		System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(output));
	}

	private Map<String, Object> convertTensorsForJson(Map<String, TensorInfo> tensors) {
		Map<String, Object> result = new LinkedHashMap<>();
		for (Map.Entry<String, TensorInfo> entry : tensors.entrySet()) {
			TensorInfo tensor = entry.getValue();
			Map<String, Object> tensorData = new LinkedHashMap<>();
			tensorData.put("shape", tensor.getShape());
			tensorData.put("type", tensor.getGgmlType().name());
			tensorData.put("offset", tensor.getOffset());
			tensorData.put("size", tensor.getSize());
			result.put(entry.getKey(), tensorData);
		}
		return result;
	}

	private void printFormattedOutput(InspectionResult result, InspectionOptions options) {
		// File information
		if (options.showFileStructure) {
			printFileInfo(result.fileInfo);
		}

		// Metadata
		if (options.showMetadata && !result.metadata.isEmpty()) {
			printMetadata(result.metadata, options);
		}

		// Tensors
		if (options.showTensors && !result.tensors.isEmpty()) {
			printTensors(result.tensors, options);
		}
	}

	private void printFileInfo(InspectionResult.FileInfo fileInfo) {
		System.out.println("=== FILE INFORMATION ===");
		System.out.printf("File: %s%n", filePath.getFileName());
		System.out.printf("Size: %,d bytes (%.2f MB)%n", fileInfo.fileSize, fileInfo.fileSize / 1024.0 / 1024.0);
		System.out.printf("Version: %d%n", fileInfo.version);
		System.out.printf("Endianness: %s (host: %s)%s%n",
			fileInfo.endianness, fileInfo.hostEndianness,
			fileInfo.endianness.equals(fileInfo.hostEndianness) ? "" : " ⚠️  MISMATCH");
		System.out.printf("Metadata entries: %d%n", fileInfo.metadataCount);
		System.out.printf("Tensors: %d%n", fileInfo.tensorCount);
		System.out.printf("SHA-256: %s%n", fileInfo.checksum);
		System.out.println();
	}

	private void printMetadata(Map<String, Object> metadata, InspectionOptions options) {
		System.out.println("=== METADATA ===");
		System.out.printf("Dumping %d key/value pairs:%n", metadata.size());

		int index = 1;
		for (Map.Entry<String, Object> entry : metadata.entrySet()) {
			String key = entry.getKey();
			Object value = entry.getValue();

			String typeName = getTypeName(value);
			String valueStr = formatValue(value, options);

			System.out.printf("%5d: %-12s | %8s | %s%n",
				index++, typeName, valueStr.length(), key);

			if (options.verbose || (value instanceof String && valueStr.length() <= options.maxStringLength)) {
				System.out.printf("       Value: %s%n", valueStr);
			}
		}
		System.out.println();
	}

	private void printTensors(Map<String, TensorInfo> tensors, InspectionOptions options) {
		System.out.println("=== TENSORS ===");
		System.out.printf("Tensor count: %d%n", tensors.size());
		System.out.printf("%-5s %-12s %-20s %-15s %-12s %s%n",
			"#", "Type", "Shape", "Size", "Offset", "Name");
		System.out.println("─".repeat(80));

		int index = 1;
		long totalSize = 0;
		for (Map.Entry<String, TensorInfo> entry : tensors.entrySet()) {
			String name = entry.getKey();
			TensorInfo tensor = entry.getValue();

			String shapeStr = Arrays.toString(tensor.getShape());
			String sizeStr = String.format("%,d", tensor.getSize());
			String offsetStr = String.format("0x%08X", tensor.getOffset());

			System.out.printf("%-5d %-12s %-20s %-15s %-12s %s%n",
				index++,
				tensor.getGgmlType().name(),
				shapeStr,
				sizeStr,
				offsetStr,
				name);

			totalSize += tensor.getSize();
		}

		System.out.println("─".repeat(80));
		System.out.printf("Total tensor data: %,d bytes (%.2f MB)%n",
			totalSize, totalSize / 1024.0 / 1024.0);
		System.out.println();
	}

	private String getTypeName(Object value) {
		if (value instanceof Number) {
			if (value instanceof Integer) return "INT32";
			if (value instanceof Long) return "INT64";
			if (value instanceof Float) return "FLOAT32";
			if (value instanceof Double) return "FLOAT64";
			if (value instanceof Byte) return "INT8";
			if (value instanceof Short) return "INT16";
		}
		if (value instanceof Boolean) return "BOOL";
		if (value instanceof String) return "STRING";
		if (value instanceof List) return "ARRAY";
		return "UNKNOWN";
	}

	private String formatValue(Object value, InspectionOptions options) {
		if (value instanceof String) {
			String str = (String) value;
			if (str.length() > options.maxStringLength && !options.verbose) {
				return str.substring(0, options.maxStringLength) + "...";
			}
			return str;
		}
		if (value instanceof List) {
			List<?> list = (List<?>) value;
			if (list.size() <= 5 || options.verbose) {
				return list.toString();
			} else {
				return String.format("[%d items: %s, %s, %s, ...]",
					list.size(), list.get(0), list.get(1), list.get(2));
			}
		}
		return value != null ? value.toString() : "null";
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
			InspectionOptions options = new InspectionOptions();
			String filePath = null;

			// Parse arguments
			for (int i = 0; i < args.length; i++) {
				switch (args[i]) {
					case "--no-metadata":
						options.metadata(false);
						break;
					case "--no-tensors":
						options.tensors(false);
						break;
					case "--no-file-info":
						options.fileStructure(false);
						break;
					case "--tensor-data":
						options.tensorData(true);
						break;
					case "--verbose":
					case "-v":
						options.verbose(true);
						break;
					case "--json":
						options.jsonOutput(true);
						break;
					case "--filter":
						if (i + 1 < args.length) {
							options.filterKey(args[++i]);
						}
						break;
					case "--max-string":
						if (i + 1 < args.length) {
							options.maxStringLength(Integer.parseInt(args[++i]));
						}
						break;
					case "--help":
					case "-h":
						printUsage();
						System.exit(0);
						break;
					default:
						if (!args[i].startsWith("-")) {
							filePath = args[i];
						}
				}
			}

			if (filePath == null) {
				System.err.println("Error: No input file specified");
				printUsage();
				System.exit(1);
			}

			GGUFInspector inspector = new GGUFInspector(Paths.get(filePath));
			inspector.printInspection(options);

		} catch (Exception e) {
			System.err.println("Error: " + e.getMessage());
			if (args.length > 0 && (Arrays.asList(args).contains("--verbose") || Arrays.asList(args).contains("-v"))) {
				e.printStackTrace();
			}
			System.exit(1);
		}
	}

	private static void printUsage() {
		System.out.println("Usage: GGUFInspector [options] <gguf_file>");
		System.out.println();
		System.out.println("Inspect GGUF file contents and structure.");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --no-metadata      Skip metadata section");
		System.out.println("  --no-tensors       Skip tensor section");
		System.out.println("  --no-file-info     Skip file information");
		System.out.println("  --tensor-data      Show tensor data content");
		System.out.println("  --verbose, -v      Show full content");
		System.out.println("  --json             Output in JSON format");
		System.out.println("  --filter <key>     Filter by key substring");
		System.out.println("  --max-string <n>   Max string length to display (default: 60)");
		System.out.println("  --help, -h         Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  GGUFInspector model.gguf");
		System.out.println("  GGUFInspector --json model.gguf > info.json");
		System.out.println("  GGUFInspector --filter \"llama\" --verbose model.gguf");
	}

	@Override
	public void close() throws IOException {
		if (reader != null) {
			reader.close();
		}
	}
}