package de.kherud.llama;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages conversation history and context for multi-turn interactions.
 * Handles chat templates, context window management, and message formatting.
 */
public class ConversationManager {

	public record Message(String role, String content, Map<String, Object> metadata) {
		public Message(String role, String content) {
			this(role, content, null);
		}

		public Message(String role, String content, Map<String, Object> metadata) {
			this.role = role;
			this.content = content;
			this.metadata = metadata != null ? new HashMap<>(metadata) : new HashMap<>();
		}


		@Override
		public Map<String, Object> metadata() {
			return Collections.unmodifiableMap(metadata);
		}
	}

	public static class ConversationConfig {
		private int maxMessages = 100;
		private int maxTokens = 4096;
		private boolean autoTruncate = true;
		private String systemPrompt = null;
		private boolean applyTemplate = true;
		private ChatTemplateFormatter.TemplateType templateType = ChatTemplateFormatter.TemplateType.CHATTML;
		private ChatTemplateFormatter.Template customTemplate = null;
		private boolean enablePersistence = false;
		private String persistenceDirectory = "conversations";
		private int maxRetries = 3;
		private long retryDelayMs = 1000;

		public ConversationConfig setMaxMessages(int maxMessages) {
			this.maxMessages = maxMessages;
			return this;
		}

		public ConversationConfig setMaxTokens(int maxTokens) {
			this.maxTokens = maxTokens;
			return this;
		}

		public ConversationConfig setAutoTruncate(boolean autoTruncate) {
			this.autoTruncate = autoTruncate;
			return this;
		}

		public ConversationConfig setSystemPrompt(String systemPrompt) {
			this.systemPrompt = systemPrompt;
			return this;
		}

		public ConversationConfig setApplyTemplate(boolean applyTemplate) {
			this.applyTemplate = applyTemplate;
			return this;
		}

		public ConversationConfig setTemplateType(ChatTemplateFormatter.TemplateType templateType) {
			this.templateType = templateType;
			return this;
		}

		public ConversationConfig setCustomTemplate(ChatTemplateFormatter.Template customTemplate) {
			this.customTemplate = customTemplate;
			return this;
		}

		public ConversationConfig setEnablePersistence(boolean enablePersistence) {
			this.enablePersistence = enablePersistence;
			return this;
		}

		public ConversationConfig setPersistenceDirectory(String persistenceDirectory) {
			this.persistenceDirectory = persistenceDirectory;
			return this;
		}

		public ConversationConfig setMaxRetries(int maxRetries) {
			this.maxRetries = maxRetries;
			return this;
		}

		public ConversationConfig setRetryDelayMs(long retryDelayMs) {
			this.retryDelayMs = retryDelayMs;
			return this;
		}

		public int getMaxRetries() {
			return maxRetries;
		}

		public long getRetryDelayMs() {
			return retryDelayMs;
		}
	}

	private final LlamaModel model;
	private final LlamaAsyncService asyncService;
	private final ConversationConfig config;
	private final List<Message> messages;
	private String conversationId;
	private final ChatTemplateFormatter templateFormatter;
	private final Map<String, Object> conversationMetrics;
	private final ObjectMapper objectMapper;

	public ConversationManager(LlamaModel model) {
		this(model, new ConversationConfig());
	}

	public ConversationManager(LlamaModel model, ConversationConfig config) {
		this.model = model;
		this.asyncService = new LlamaAsyncService(model);
		this.config = config;
		this.messages = new ArrayList<>();
		this.conversationId = generateConversationId();
		this.conversationMetrics = new ConcurrentHashMap<>();
		this.objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

		// Initialize template formatter
		if (config.customTemplate != null) {
			this.templateFormatter = new ChatTemplateFormatter(config.customTemplate);
		} else {
			this.templateFormatter = new ChatTemplateFormatter(config.templateType);
		}

		if (config.systemPrompt != null) {
			messages.add(new Message("system", config.systemPrompt));
		}

		// Initialize metrics
		conversationMetrics.put("totalTokensGenerated", 0);
		conversationMetrics.put("totalTokensPrompt", 0);
		conversationMetrics.put("totalInferenceTimeMs", 0L);
		conversationMetrics.put("messageCount", 0);
		conversationMetrics.put("startTime", System.currentTimeMillis());

		// Create persistence directory if enabled
		if (config.enablePersistence) {
			new File(config.persistenceDirectory).mkdirs();
		}
	}

	/**
	 * Add a user message and generate an assistant response
	 */
	public String chat(String userMessage) {
		return chat(userMessage, new InferenceParameters(""));
	}

	/**
	 * Add a user message and generate an assistant response with custom parameters
	 */
	public String chat(String userMessage, InferenceParameters parameters) {
		// Add user message
		addMessage("user", userMessage);

		// Build prompt from conversation history
		String prompt = buildPrompt();
		parameters.setPrompt(prompt);

		// Generate response with retry logic
		String response = null;
		int attempts = 0;
		long startTime = System.currentTimeMillis();

		while (attempts < config.maxRetries) {
			try {
				response = model.complete(parameters);
				break;
			} catch (Exception e) {
				attempts++;
				if (attempts >= config.maxRetries) {
					throw new RuntimeException("Failed after " + config.maxRetries + " attempts", e);
				}
				try {
					Thread.sleep(config.retryDelayMs * attempts);
				} catch (InterruptedException ie) {
					Thread.currentThread().interrupt();
					throw new RuntimeException("Interrupted during retry", ie);
				}
			}
		}

		long inferenceTime = System.currentTimeMillis() - startTime;

		// Add assistant response
		addMessage("assistant", response);

		// Update metrics
		updateMetrics(prompt, response, inferenceTime);

		// Manage context window
		if (config.autoTruncate) {
			truncateIfNeeded();
		}

		// Persist if enabled
		if (config.enablePersistence) {
			persistConversation();
		}

		return response;
	}

	/**
	 * Async version of chat
	 */
	public CompletableFuture<String> chatAsync(String userMessage) {
		return chatAsync(userMessage, new InferenceParameters(""));
	}

	/**
	 * Async version of chat with custom parameters
	 */
	public CompletableFuture<String> chatAsync(String userMessage, InferenceParameters parameters) {
		// Add user message
		addMessage("user", userMessage);

		// Build prompt from conversation history
		String prompt = buildPrompt();
		parameters.setPrompt(prompt);

		// Generate response asynchronously
		return asyncService.completeAsync(parameters).thenApply(response -> {
			// Add assistant response
			addMessage("assistant", response);

			// Manage context window
			if (config.autoTruncate) {
				truncateIfNeeded();
			}

			return response;
		});
	}

	/**
	 * Stream chat response
	 */
	public Iterable<LlamaOutput> streamChat(String userMessage, InferenceParameters parameters) {
		// Add user message
		addMessage("user", userMessage);

		// Build prompt from conversation history
		String prompt = buildPrompt();
		parameters.setPrompt(prompt);

		// Stream response
		StringBuilder responseBuilder = new StringBuilder();

		return () -> {
			var iterator = model.generate(parameters).iterator();
			return new java.util.Iterator<LlamaOutput>() {
				@Override
				public boolean hasNext() {
					return iterator.hasNext();
				}

				@Override
				public LlamaOutput next() {
					LlamaOutput output = iterator.next();
					responseBuilder.append(output.text);

					// If this is the last token, add the complete response to history
					if (!iterator.hasNext()) {
						addMessage("assistant", responseBuilder.toString());
						if (config.autoTruncate) {
							truncateIfNeeded();
						}
					}

					return output;
				}
			};
		};
	}

	/**
	 * Add a message to the conversation history
	 */
	public void addMessage(String role, String content) {
		messages.add(new Message(role, content));
	}

	/**
	 * Add a message with metadata
	 */
	public void addMessage(String role, String content, Map<String, Object> metadata) {
		messages.add(new Message(role, content, metadata));
	}

	/**
	 * Get conversation history
	 */
	public List<Message> getMessages() {
		return Collections.unmodifiableList(messages);
	}

	/**
	 * Clear conversation history (keeps system prompt if configured)
	 */
	public void clear() {
		messages.clear();
		if (config.systemPrompt != null) {
			messages.add(new Message("system", config.systemPrompt));
		}
		conversationId = generateConversationId();
	}

	/**
	 * Build prompt from conversation history
	 */
	private String buildPrompt() {
		if (config.applyTemplate) {
			// Use proper template formatting
			return templateFormatter.format(messages);
		} else {
			// Simple concatenation without role formatting
			StringBuilder prompt = new StringBuilder();
			for (Message msg : messages) {
				prompt.append(msg.content()).append("\n");
			}
			return prompt.toString();
		}
	}

	/**
	 * Truncate conversation history if needed
	 */
	private void truncateIfNeeded() {
		// Keep system message if present
		int startIdx = (config.systemPrompt != null) ? 1 : 0;

		// Truncate by message count
		while (messages.size() > config.maxMessages + startIdx) {
			messages.remove(startIdx);
		}

		// Truncate by token count (approximate)
		if (config.maxTokens > 0) {
			int totalTokens = countTokensApprox();
			while (totalTokens > config.maxTokens && messages.size() > startIdx + 2) {
				messages.remove(startIdx);
				totalTokens = countTokensApprox();
			}
		}
	}

	/**
	 * Count tokens using model's tokenizer
	 */
	private int countTokensApprox() {
		// If model supports tokenization, use it
		try {
			String fullPrompt = buildPrompt();
			int[] tokens = model.encode(fullPrompt);
			return tokens.length;
		} catch (Exception e) {
			// Fallback to approximation
			int totalChars = 0;
			for (Message msg : messages) {
				totalChars += msg.content.length();
			}
			// Rough approximation: 4 chars per token
			return totalChars / 4;
		}
	}

	/**
	 * Generate unique conversation ID
	 */
	private static String generateConversationId() {
		return "conv-" + System.currentTimeMillis() + "-" + Thread.currentThread().getId() + "-" + Math.random();
	}

	/**
	 * Get conversation ID
	 */
	public String getConversationId() {
		return conversationId;
	}

	/**
	 * Export conversation as JSON-friendly format
	 */
	public Map<String, Object> export() {
		Map<String, Object> export = new HashMap<>();
		export.put("conversationId", conversationId);
		export.put("messageCount", messages.size());

		List<Map<String, Object>> messageList = new ArrayList<>();
		for (Message msg : messages) {
			Map<String, Object> msgMap = new HashMap<>();
			msgMap.put("role", msg.role);
			msgMap.put("content", msg.content);
			if (!msg.metadata.isEmpty()) {
				msgMap.put("metadata", msg.metadata);
			}
			messageList.add(msgMap);
		}
		export.put("messages", messageList);

		return export;
	}

	/**
	 * Import conversation from exported format
	 */
	public void importConversation(Map<String, Object> data) {
		messages.clear();

		@SuppressWarnings("unchecked")
		List<Map<String, Object>> messageList = (List<Map<String, Object>>) data.get("messages");

		for (Map<String, Object> msgMap : messageList) {
			String role = (String) msgMap.get("role");
			String content = (String) msgMap.get("content");
			@SuppressWarnings("unchecked")
			Map<String, Object> metadata = (Map<String, Object>) msgMap.get("metadata");

			messages.add(new Message(role, content, metadata));
		}

		if (data.containsKey("conversationId")) {
			conversationId = (String) data.get("conversationId");
		}

		if (data.containsKey("metrics")) {
			@SuppressWarnings("unchecked")
			Map<String, Object> metrics = (Map<String, Object>) data.get("metrics");
			conversationMetrics.putAll(metrics);
		}
	}

	/**
	 * Update conversation metrics
	 */
	private void updateMetrics(String prompt, String response, long inferenceTime) {
		try {
			int promptTokens = model.encode(prompt).length;
			int responseTokens = model.encode(response).length;

			int totalPromptTokens = (int) conversationMetrics.getOrDefault("totalTokensPrompt", 0);
			int totalGenerated = (int) conversationMetrics.getOrDefault("totalTokensGenerated", 0);
			long totalTime = (long) conversationMetrics.getOrDefault("totalInferenceTimeMs", 0L);

			conversationMetrics.put("totalTokensPrompt", totalPromptTokens + promptTokens);
			conversationMetrics.put("totalTokensGenerated", totalGenerated + responseTokens);
			conversationMetrics.put("totalInferenceTimeMs", totalTime + inferenceTime);
			conversationMetrics.put("messageCount", messages.size());
			conversationMetrics.put("lastUpdateTime", System.currentTimeMillis());

			// Calculate averages
			if (totalGenerated > 0) {
				double tokensPerSecond = (totalGenerated / (totalTime / 1000.0));
				conversationMetrics.put("averageTokensPerSecond", tokensPerSecond);
			}
		} catch (Exception e) {
			// Metrics are optional, don't fail on error
		}
	}

	/**
	 * Get conversation metrics
	 */
	public Map<String, Object> getMetrics() {
		return Collections.unmodifiableMap(conversationMetrics);
	}

	/**
	 * Persist conversation to disk
	 */
	private void persistConversation() {
		if (!config.enablePersistence) return;

		try {
			File file = new File(config.persistenceDirectory, conversationId + ".json");
			Map<String, Object> data = export();
			data.put("metrics", conversationMetrics);
			data.put("templateType", config.templateType.toString());

			try (FileWriter writer = new FileWriter(file)) {
				objectMapper.writeValue(writer, data);
			}
		} catch (IOException e) {
			// Log error but don't fail
		}
	}

	/**
	 * Load conversation from disk
	 */
	public static ConversationManager loadConversation(LlamaModel model, String conversationId, String directory) {
		try {
			File file = new File(directory, conversationId + ".json");
			if (!file.exists()) {
				return null;
			}

			ObjectMapper mapper = new ObjectMapper();
			try (FileReader reader = new FileReader(file)) {
				@SuppressWarnings("unchecked")
				Map<String, Object> data = mapper.readValue(reader, Map.class);

				// Rebuild config
				ConversationConfig config = new ConversationConfig();
				config.setEnablePersistence(true);
				config.setPersistenceDirectory(directory);

				if (data.containsKey("templateType")) {
					String templateType = (String) data.get("templateType");
					config.setTemplateType(ChatTemplateFormatter.TemplateType.fromString(templateType));
				}

				ConversationManager manager = new ConversationManager(model, config);
				manager.importConversation(data);
				return manager;
			}
		} catch (IOException e) {
			return null;
		}
	}

	/**
	 * Fork the conversation at current point
	 */
	public ConversationManager fork() {
		ConversationManager forked = new ConversationManager(model, config);
		forked.messages.clear();
		forked.messages.addAll(messages);
		forked.conversationId = generateConversationId();
		return forked;
	}

	/**
	 * Get template formatter
	 */
	public ChatTemplateFormatter getTemplateFormatter() {
		return templateFormatter;
	}
}
