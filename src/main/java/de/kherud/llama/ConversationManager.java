package de.kherud.llama;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

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
	}

	private final LlamaModel model;
	private final LlamaAsyncService asyncService;
	private final ConversationConfig config;
	private final List<Message> messages;
	private String conversationId;

	public ConversationManager(LlamaModel model) {
		this(model, new ConversationConfig());
	}

	public ConversationManager(LlamaModel model, ConversationConfig config) {
		this.model = model;
		this.asyncService = new LlamaAsyncService(model);
		this.config = config;
		this.messages = new ArrayList<>();
		this.conversationId = generateConversationId();

		if (config.systemPrompt != null) {
			messages.add(new Message("system", config.systemPrompt));
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

		// Generate response
		String response = model.complete(parameters);

		// Add assistant response
		addMessage("assistant", response);

		// Manage context window
		if (config.autoTruncate) {
			truncateIfNeeded();
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
			// Use formatted template approach
			StringBuilder prompt = new StringBuilder();
			for (Message msg : messages) {
				// Apply basic role-based formatting when templates are enabled
				prompt.append(msg.role()).append(": ").append(msg.content()).append("\n");
			}
			return prompt.toString();
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
	 * Approximate token count (can be replaced with actual tokenization)
	 */
	private int countTokensApprox() {
		int totalChars = 0;
		for (Message msg : messages) {
			totalChars += msg.content.length();
		}
		// Rough approximation: 4 chars per token
		return totalChars / 4;
	}

	/**
	 * Generate unique conversation ID
	 */
	private static String generateConversationId() {
		return "conv-" + System.currentTimeMillis() + "-" + Thread.currentThread().getId();
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
	}
}
