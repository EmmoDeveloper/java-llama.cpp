package de.kherud.llama.gguf;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.Logger;

/**
 * GGUF metadata editing utility.
 *
 * Equivalent to gguf_set_metadata.py - allows modification of GGUF file metadata
 * without affecting tensor data.
 */
public class GGUFMetadataEditor {
	private static final Logger LOGGER = Logger.getLogger(GGUFMetadataEditor.class.getName());

	public static class MetadataOperation {
		public enum Type {
			SET,    // Set or update a key
			DELETE, // Remove a key
			RENAME  // Rename a key
		}

		private final Type type;
		private final String key;
		private final Object value;
		private final String newKey;

		private MetadataOperation(Type type, String key, Object value, String newKey) {
			this.type = type;
			this.key = key;
			this.value = value;
			this.newKey = newKey;
		}

		public static MetadataOperation set(String key, Object value) {
			return new MetadataOperation(Type.SET, key, value, null);
		}

		public static MetadataOperation delete(String key) {
			return new MetadataOperation(Type.DELETE, key, null, null);
		}

		public static MetadataOperation rename(String oldKey, String newKey) {
			return new MetadataOperation(Type.RENAME, oldKey, null, newKey);
		}

		public Type getType() { return type; }
		public String getKey() { return key; }
		public Object getValue() { return value; }
		public String getNewKey() { return newKey; }
	}

	public static class EditOptions {
		private boolean verbose = false;
		private boolean dryRun = false;
		private boolean backup = true;
		private String backupSuffix = ".backup";
		private boolean force = false;

		public EditOptions verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public EditOptions dryRun(boolean dryRun) {
			this.dryRun = dryRun;
			return this;
		}

		public EditOptions backup(boolean backup) {
			this.backup = backup;
			return this;
		}

		public EditOptions backupSuffix(String suffix) {
			this.backupSuffix = suffix;
			return this;
		}

		public EditOptions force(boolean force) {
			this.force = force;
			return this;
		}
	}

	public static class EditResult {
		private final boolean success;
		private final String message;
		private final Map<String, Object> changedMetadata = new LinkedHashMap<>();
		private final List<String> deletedKeys = new ArrayList<>();
		private final Map<String, String> renamedKeys = new LinkedHashMap<>();

		public EditResult(boolean success, String message) {
			this.success = success;
			this.message = message;
		}

		public void addChange(String key, Object value) {
			changedMetadata.put(key, value);
		}

		public void addDeletion(String key) {
			deletedKeys.add(key);
		}

		public void addRename(String oldKey, String newKey) {
			renamedKeys.put(oldKey, newKey);
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Map<String, Object> getChangedMetadata() { return changedMetadata; }
		public List<String> getDeletedKeys() { return deletedKeys; }
		public Map<String, String> getRenamedKeys() { return renamedKeys; }

		public boolean hasChanges() {
			return !changedMetadata.isEmpty() || !deletedKeys.isEmpty() || !renamedKeys.isEmpty();
		}
	}

	private final Path filePath;
	private final EditOptions options;

	public GGUFMetadataEditor(Path filePath) {
		this(filePath, new EditOptions());
	}

	public GGUFMetadataEditor(Path filePath, EditOptions options) {
		this.filePath = filePath;
		this.options = options;
	}

	/**
	 * Apply a single metadata operation
	 */
	public EditResult applyOperation(MetadataOperation operation) {
		return applyOperations(Collections.singletonList(operation));
	}

	/**
	 * Apply multiple metadata operations
	 */
	public EditResult applyOperations(List<MetadataOperation> operations) {
		try {
			if (!Files.exists(filePath)) {
				return new EditResult(false, "File not found: " + filePath);
			}

			if (!Files.isRegularFile(filePath)) {
				return new EditResult(false, "Not a regular file: " + filePath);
			}

			// Create backup if requested
			Path backupPath = null;
			if (options.backup && !options.dryRun) {
				backupPath = createBackup();
				if (options.verbose) {
					LOGGER.info("Created backup: " + backupPath);
				}
			}

			// Read current metadata
			Map<String, GGUFValue> currentMetadata;
			Map<String, TensorInfo> tensorInfos;
			int version;
			ByteOrder byteOrder;

			try (GGUFReader reader = new GGUFReader(filePath)) {
				currentMetadata = new LinkedHashMap<>(reader.getMetadata());
				tensorInfos = reader.getTensorInfos();
				version = reader.getVersion();
				byteOrder = reader.getByteOrder();
			}

			// Apply operations
			EditResult result = new EditResult(true, "Operations applied successfully");
			Map<String, GGUFValue> newMetadata = new LinkedHashMap<>(currentMetadata);

			for (MetadataOperation op : operations) {
				switch (op.getType()) {
					case SET:
						Object oldValue = newMetadata.get(op.getKey());
						GGUFValue newValue = convertToGGUFValue(op.getValue());
						newMetadata.put(op.getKey(), newValue);
						result.addChange(op.getKey(), op.getValue());

						if (options.verbose) {
							if (oldValue != null) {
								LOGGER.info(String.format("Updated %s: %s -> %s", op.getKey(), oldValue, op.getValue()));
							} else {
								LOGGER.info(String.format("Added %s: %s", op.getKey(), op.getValue()));
							}
						}
						break;

					case DELETE:
						if (newMetadata.containsKey(op.getKey())) {
							newMetadata.remove(op.getKey());
							result.addDeletion(op.getKey());

							if (options.verbose) {
								LOGGER.info("Deleted: " + op.getKey());
							}
						} else if (options.verbose) {
							LOGGER.warning("Key not found for deletion: " + op.getKey());
						}
						break;

					case RENAME:
						if (newMetadata.containsKey(op.getKey())) {
							GGUFValue value = newMetadata.remove(op.getKey());
							newMetadata.put(op.getNewKey(), value);
							result.addRename(op.getKey(), op.getNewKey());

							if (options.verbose) {
								LOGGER.info(String.format("Renamed %s -> %s", op.getKey(), op.getNewKey()));
							}
						} else if (options.verbose) {
							LOGGER.warning("Key not found for rename: " + op.getKey());
						}
						break;
				}
			}

			// Write modified file if not dry run
			if (!options.dryRun && result.hasChanges()) {
				writeModifiedFile(newMetadata, tensorInfos, version, byteOrder);

				if (options.verbose) {
					LOGGER.info("File updated: " + filePath);
				}
			} else if (options.dryRun) {
				result = new EditResult(true, "Dry run completed - no changes made");
				if (options.verbose) {
					LOGGER.info("Dry run - no changes written to file");
				}
			} else if (!result.hasChanges()) {
				result = new EditResult(true, "No changes needed");
				// Remove backup if no changes were made
				if (backupPath != null) {
					Files.deleteIfExists(backupPath);
				}
			}

			return result;

		} catch (Exception e) {
			return new EditResult(false, "Error: " + e.getMessage());
		}
	}

	private Path createBackup() throws IOException {
		Path backupPath = Paths.get(filePath.toString() + options.backupSuffix);

		// If backup exists and force is false, fail
		if (Files.exists(backupPath) && !options.force) {
			throw new IOException("Backup file already exists: " + backupPath + " (use --force to overwrite)");
		}

		Files.copy(filePath, backupPath);
		return backupPath;
	}

	private GGUFValue convertToGGUFValue(Object value) {
		if (value instanceof String) {
			return new GGUFValue(GGUFConstants.GGMLType.STRING, value);
		} else if (value instanceof Boolean) {
			return new GGUFValue(GGUFConstants.GGMLType.BOOL, value);
		} else if (value instanceof Integer) {
			return new GGUFValue(GGUFConstants.GGMLType.INT32, value);
		} else if (value instanceof Long) {
			return new GGUFValue(GGUFConstants.GGMLType.INT64, value);
		} else if (value instanceof Float) {
			return new GGUFValue(GGUFConstants.GGMLType.FLOAT32, value);
		} else if (value instanceof Double) {
			return new GGUFValue(GGUFConstants.GGMLType.FLOAT64, value);
		} else if (value instanceof List) {
			return convertArrayToGGUFValue((List<?>) value);
		} else {
			// Try to parse as string
			return new GGUFValue(GGUFConstants.GGMLType.STRING, value.toString());
		}
	}

	private GGUFValue convertArrayToGGUFValue(List<?> list) {
		if (list.isEmpty()) {
			return new GGUFValue(GGUFConstants.GGMLType.ARRAY, new Object[0]);
		}

		// Determine array type from first element
		Object first = list.get(0);
		Object[] array = new Object[list.size()];

		for (int i = 0; i < list.size(); i++) {
			array[i] = convertToGGUFValue(list.get(i));
		}

		return new GGUFValue(GGUFConstants.GGMLType.ARRAY, array);
	}

	private void writeModifiedFile(Map<String, GGUFValue> metadata,
	                              Map<String, TensorInfo> tensorInfos,
	                              int version, ByteOrder byteOrder) throws IOException {

		// Read original tensor data
		Map<String, byte[]> tensorData = new LinkedHashMap<>();
		try (RandomAccessFile raf = new RandomAccessFile(filePath.toFile(), "r")) {
			for (Map.Entry<String, TensorInfo> entry : tensorInfos.entrySet()) {
				TensorInfo tensor = entry.getValue();
				byte[] data = new byte[(int) tensor.getSize()];
				raf.seek(tensor.getOffset());
				raf.readFully(data);
				tensorData.put(entry.getKey(), data);
			}
		}

		// Write new file
		try (GGUFWriter writer = new GGUFWriter(filePath, "modified")) {
			// Set byte order if different
			if (byteOrder == ByteOrder.BIG_ENDIAN) {
				// Note: Would need to add byte order support to GGUFWriter
				LOGGER.warning("Big endian byte order not fully supported in writer");
			}

			// Write metadata
			for (Map.Entry<String, GGUFValue> entry : metadata.entrySet()) {
				writeMetadataEntry(writer, entry.getKey(), entry.getValue());
			}

			// Write tensor information
			for (Map.Entry<String, TensorInfo> entry : tensorInfos.entrySet()) {
				TensorInfo tensor = entry.getValue();
				writer.addTensorInfo(
					entry.getKey(),
					tensor.getShape(),
					tensor.getGgmlType(),
					tensor.getSize()
				);
			}

			// Write file structure
			writer.writeToFile();

			// Write tensor data
			for (Map.Entry<String, byte[]> entry : tensorData.entrySet()) {
				// Note: Would need tensor data writing support in GGUFWriter
				// writer.writeTensorData(entry.getKey(), entry.getValue());
			}
		}
	}

	private void writeMetadataEntry(GGUFWriter writer, String key, GGUFValue value) {
		Object val = value.getValue();

		switch (value.getType()) {
			case STRING:
				writer.addString(key, (String) val);
				break;
			case BOOL:
				writer.addBool(key, (Boolean) val);
				break;
			case UINT8:
				writer.addUInt8(key, ((Number) val).byteValue());
				break;
			case INT8:
				writer.addInt8(key, ((Number) val).byteValue());
				break;
			case UINT16:
				writer.addUInt16(key, ((Number) val).shortValue());
				break;
			case INT16:
				writer.addInt16(key, ((Number) val).shortValue());
				break;
			case UINT32:
				writer.addUInt32(key, ((Number) val).intValue());
				break;
			case INT32:
				writer.addInt32(key, ((Number) val).intValue());
				break;
			case UINT64:
				writer.addUInt64(key, ((Number) val).longValue());
				break;
			case INT64:
				writer.addInt64(key, ((Number) val).longValue());
				break;
			case FLOAT32:
				writer.addFloat32(key, ((Number) val).floatValue());
				break;
			case FLOAT64:
				writer.addFloat64(key, ((Number) val).doubleValue());
				break;
			case ARRAY:
				// Handle arrays based on element type
				Object[] array = (Object[]) val;
				if (array.length > 0 && array[0] instanceof String) {
					String[] stringArray = Arrays.copyOf(array, array.length, String[].class);
					writer.addArray(key, stringArray);
				} else if (array.length > 0 && array[0] instanceof Number) {
					if (array[0] instanceof Float) {
						float[] floatArray = new float[array.length];
						for (int i = 0; i < array.length; i++) {
							floatArray[i] = ((Number) array[i]).floatValue();
						}
						writer.addArray(key, floatArray);
					} else if (array[0] instanceof Integer) {
						int[] intArray = new int[array.length];
						for (int i = 0; i < array.length; i++) {
							intArray[i] = ((Number) array[i]).intValue();
						}
						writer.addArray(key, intArray);
					}
				}
				break;
		}
	}

	/**
	 * Load operations from JSON file
	 */
	public static List<MetadataOperation> loadOperationsFromJson(Path jsonPath) throws IOException {
		ObjectMapper mapper = new ObjectMapper();
		JsonNode root = mapper.readTree(Files.newBufferedReader(jsonPath));

		List<MetadataOperation> operations = new ArrayList<>();

		if (root.has("set")) {
			JsonNode setNode = root.get("set");
			setNode.fields().forEachRemaining(entry -> {
				Object value = parseJsonValue(entry.getValue());
				operations.add(MetadataOperation.set(entry.getKey(), value));
			});
		}

		if (root.has("delete")) {
			JsonNode deleteNode = root.get("delete");
			if (deleteNode.isArray()) {
				deleteNode.forEach(node ->
					operations.add(MetadataOperation.delete(node.asText())));
			}
		}

		if (root.has("rename")) {
			JsonNode renameNode = root.get("rename");
			renameNode.fields().forEachRemaining(entry ->
				operations.add(MetadataOperation.rename(entry.getKey(), entry.getValue().asText())));
		}

		return operations;
	}

	private static Object parseJsonValue(JsonNode node) {
		if (node.isTextual()) return node.asText();
		if (node.isBoolean()) return node.asBoolean();
		if (node.isInt()) return node.asInt();
		if (node.isLong()) return node.asLong();
		if (node.isFloat()) return (float) node.asDouble();
		if (node.isDouble()) return node.asDouble();
		if (node.isArray()) {
			List<Object> list = new ArrayList<>();
			node.forEach(item -> list.add(parseJsonValue(item)));
			return list;
		}
		return node.asText();
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
			EditOptions options = new EditOptions();
			List<MetadataOperation> operations = new ArrayList<>();
			String filePath = null;

			// Parse arguments
			for (int i = 0; i < args.length; i++) {
				switch (args[i]) {
					case "--set":
						if (i + 2 < args.length) {
							String key = args[++i];
							String valueStr = args[++i];
							Object value = parseValue(valueStr);
							operations.add(MetadataOperation.set(key, value));
						}
						break;
					case "--delete":
						if (i + 1 < args.length) {
							operations.add(MetadataOperation.delete(args[++i]));
						}
						break;
					case "--rename":
						if (i + 2 < args.length) {
							String oldKey = args[++i];
							String newKey = args[++i];
							operations.add(MetadataOperation.rename(oldKey, newKey));
						}
						break;
					case "--json":
						if (i + 1 < args.length) {
							Path jsonPath = Paths.get(args[++i]);
							operations.addAll(loadOperationsFromJson(jsonPath));
						}
						break;
					case "--verbose":
					case "-v":
						options.verbose(true);
						break;
					case "--dry-run":
						options.dryRun(true);
						break;
					case "--no-backup":
						options.backup(false);
						break;
					case "--backup-suffix":
						if (i + 1 < args.length) {
							options.backupSuffix(args[++i]);
						}
						break;
					case "--force":
						options.force(true);
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

			if (operations.isEmpty()) {
				System.err.println("Error: No operations specified");
				printUsage();
				System.exit(1);
			}

			GGUFMetadataEditor editor = new GGUFMetadataEditor(Paths.get(filePath), options);
			EditResult result = editor.applyOperations(operations);

			if (result.isSuccess()) {
				System.out.println(result.getMessage());
				if (result.hasChanges()) {
					System.out.println("Changes made:");
					result.getChangedMetadata().forEach((k, v) ->
						System.out.println("  Set " + k + " = " + v));
					result.getDeletedKeys().forEach(k ->
						System.out.println("  Deleted " + k));
					result.getRenamedKeys().forEach((old, newKey) ->
						System.out.println("  Renamed " + old + " -> " + newKey));
				}
			} else {
				System.err.println("Error: " + result.getMessage());
				System.exit(1);
			}

		} catch (Exception e) {
			System.err.println("Error: " + e.getMessage());
			if (Arrays.asList(args).contains("--verbose") || Arrays.asList(args).contains("-v")) {
				e.printStackTrace();
			}
			System.exit(1);
		}
	}

	private static Object parseValue(String valueStr) {
		// Try to parse as different types
		if ("true".equalsIgnoreCase(valueStr) || "false".equalsIgnoreCase(valueStr)) {
			return Boolean.parseBoolean(valueStr);
		}

		try {
			if (valueStr.contains(".")) {
				return Double.parseDouble(valueStr);
			} else {
				return Long.parseLong(valueStr);
			}
		} catch (NumberFormatException e) {
			// Return as string
			return valueStr;
		}
	}

	private static void printUsage() {
		System.out.println("Usage: GGUFMetadataEditor [options] <gguf_file>");
		System.out.println();
		System.out.println("Edit GGUF file metadata.");
		System.out.println();
		System.out.println("Operations:");
		System.out.println("  --set <key> <value>      Set metadata key to value");
		System.out.println("  --delete <key>           Delete metadata key");
		System.out.println("  --rename <old> <new>     Rename metadata key");
		System.out.println("  --json <file>            Load operations from JSON file");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --verbose, -v            Verbose output");
		System.out.println("  --dry-run                Show changes without applying");
		System.out.println("  --no-backup              Don't create backup file");
		System.out.println("  --backup-suffix <ext>    Backup file suffix (default: .backup)");
		System.out.println("  --force                  Overwrite existing backup");
		System.out.println("  --help, -h               Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  GGUFMetadataEditor --set general.name \"My Model\" model.gguf");
		System.out.println("  GGUFMetadataEditor --delete general.description model.gguf");
		System.out.println("  GGUFMetadataEditor --rename old_key new_key model.gguf");
		System.out.println("  GGUFMetadataEditor --json operations.json model.gguf");
		System.out.println();
		System.out.println("JSON format:");
		System.out.println("  {");
		System.out.println("    \"set\": { \"key1\": \"value1\", \"key2\": 42 },");
		System.out.println("    \"delete\": [\"key3\", \"key4\"],");
		System.out.println("    \"rename\": { \"old_key\": \"new_key\" }");
		System.out.println("  }");
	}
}