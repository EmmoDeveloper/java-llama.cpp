package de.kherud.llama;

import java.util.List;
import java.util.Map;

/**
 * Formats conversation messages according to various chat templates.
 * Supports Llama, ChatML, Alpaca, and custom templates.
 */
public class ChatTemplateFormatter {

	public enum TemplateType {
		LLAMA3("llama3"),
		CHATTML("chatml"),
		ALPACA("alpaca"),
		LLAMA2("llama2"),
		MISTRAL("mistral"),
		ZEPHYR("zephyr"),
		VICUNA("vicuna"),
		CUSTOM("custom");

		private final String name;

		TemplateType(String name) {
			this.name = name;
		}

		public String getName() {
			return name;
		}

		public static TemplateType fromString(String name) {
			for (TemplateType type : values()) {
				if (type.name.equalsIgnoreCase(name)) {
					return type;
				}
			}
			return CUSTOM;
		}
	}

	public static class Template {
		private final String systemOpen;
		private final String systemClose;
		private final String userOpen;
		private final String userClose;
		private final String assistantOpen;
		private final String assistantClose;
		private final String eosToken;
		private final String bosToken;
		private final boolean addGenerationPrompt;

		public Template(String systemOpen, String systemClose, String userOpen, String userClose,
						String assistantOpen, String assistantClose, String eosToken, String bosToken,
						boolean addGenerationPrompt) {
			this.systemOpen = systemOpen;
			this.systemClose = systemClose;
			this.userOpen = userOpen;
			this.userClose = userClose;
			this.assistantOpen = assistantOpen;
			this.assistantClose = assistantClose;
			this.eosToken = eosToken;
			this.bosToken = bosToken;
			this.addGenerationPrompt = addGenerationPrompt;
		}

		// Builder pattern for custom templates
		public static class Builder {
			private String systemOpen = "";
			private String systemClose = "";
			private String userOpen = "";
			private String userClose = "";
			private String assistantOpen = "";
			private String assistantClose = "";
			private String eosToken = "";
			private String bosToken = "";
			private boolean addGenerationPrompt = true;

			public Builder setSystemTags(String open, String close) {
				this.systemOpen = open;
				this.systemClose = close;
				return this;
			}

			public Builder setUserTags(String open, String close) {
				this.userOpen = open;
				this.userClose = close;
				return this;
			}

			public Builder setAssistantTags(String open, String close) {
				this.assistantOpen = open;
				this.assistantClose = close;
				return this;
			}

			public Builder setSpecialTokens(String bos, String eos) {
				this.bosToken = bos;
				this.eosToken = eos;
				return this;
			}

			public Builder setAddGenerationPrompt(boolean add) {
				this.addGenerationPrompt = add;
				return this;
			}

			public Template build() {
				return new Template(systemOpen, systemClose, userOpen, userClose,
					assistantOpen, assistantClose, eosToken, bosToken, addGenerationPrompt);
			}
		}
	}

	private static final Map<TemplateType, Template> TEMPLATES = Map.of(
		TemplateType.LLAMA3, new Template(
			"<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>",
			"<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>",
			"<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>",
			"<|eot_id|>", "<|begin_of_text|>", true
		),
		TemplateType.CHATTML, new Template(
			"<|im_start|>system\n", "<|im_end|>\n",
			"<|im_start|>user\n", "<|im_end|>\n",
			"<|im_start|>assistant\n", "<|im_end|>\n",
			"<|im_end|>", "", true
		),
		TemplateType.ALPACA, new Template(
			"", "",
			"### Instruction:\n", "\n",
			"### Response:\n", "\n",
			"", "", true
		),
		TemplateType.LLAMA2, new Template(
			"<<SYS>>\n", "\n<</SYS>>\n\n",
			"[INST] ", " [/INST]\n",
			"", "\n",
			"</s>", "<s>", true
		),
		TemplateType.MISTRAL, new Template(
			"", "",
			"[INST] ", " [/INST]",
			"", "</s>",
			"</s>", "<s>", true
		),
		TemplateType.ZEPHYR, new Template(
			"<|system|>\n", "</s>\n",
			"<|user|>\n", "</s>\n",
			"<|assistant|>\n", "</s>\n",
			"</s>", "", true
		),
		TemplateType.VICUNA, new Template(
			"", "",
			"USER: ", "\n",
			"ASSISTANT: ", "\n",
			"", "", true
		)
	);

	private final TemplateType type;
	private final Template template;

	public ChatTemplateFormatter(TemplateType type) {
		this.type = type;
		this.template = TEMPLATES.getOrDefault(type, TEMPLATES.get(TemplateType.CHATTML));
	}

	public ChatTemplateFormatter(Template customTemplate) {
		this.type = TemplateType.CUSTOM;
		this.template = customTemplate;
	}

	/**
	 * Format a list of messages according to the template
	 */
	public String format(List<ConversationManager.Message> messages) {
		StringBuilder formatted = new StringBuilder();

		// Add BOS token if needed
		if (!template.bosToken.isEmpty()) {
			formatted.append(template.bosToken);
		}

		boolean hasSystem = false;
		for (ConversationManager.Message message : messages) {
			String role = message.role().toLowerCase();
			String content = message.content();

			switch (role) {
				case "system":
					formatted.append(template.systemOpen);
					formatted.append(content);
					formatted.append(template.systemClose);
					hasSystem = true;
					break;
				case "user":
					// For Llama2, combine system message with first user message
					if (type == TemplateType.LLAMA2 && hasSystem && isFirstUserMessage(messages, message)) {
						// Already handled in system case
					} else {
						formatted.append(template.userOpen);
						formatted.append(content);
						formatted.append(template.userClose);
					}
					break;
				case "assistant":
					formatted.append(template.assistantOpen);
					formatted.append(content);
					formatted.append(template.assistantClose);
					break;
				case "function":
					// Handle function responses as assistant messages
					formatted.append(template.assistantOpen);
					formatted.append("[Function Result] ");
					formatted.append(content);
					formatted.append(template.assistantClose);
					break;
				default:
					// Unknown role, treat as user
					formatted.append(template.userOpen);
					formatted.append(content);
					formatted.append(template.userClose);
					break;
			}
		}

		// Add generation prompt if the last message was from user
		if (template.addGenerationPrompt && !messages.isEmpty()) {
			ConversationManager.Message lastMessage = messages.get(messages.size() - 1);
			if ("user".equals(lastMessage.role().toLowerCase())) {
				formatted.append(template.assistantOpen);
			}
		}

		return formatted.toString();
	}

	/**
	 * Format with explicit system prompt override
	 */
	public String formatWithSystem(String systemPrompt, List<ConversationManager.Message> messages) {
		StringBuilder formatted = new StringBuilder();

		// Add BOS token if needed
		if (!template.bosToken.isEmpty()) {
			formatted.append(template.bosToken);
		}

		// Add system prompt
		if (systemPrompt != null && !systemPrompt.isEmpty()) {
			formatted.append(template.systemOpen);
			formatted.append(systemPrompt);
			formatted.append(template.systemClose);
		}

		// Add rest of messages, skipping any system messages
		for (ConversationManager.Message message : messages) {
			if (!"system".equals(message.role().toLowerCase())) {
				String role = message.role().toLowerCase();
				String content = message.content();

				switch (role) {
					case "user":
						formatted.append(template.userOpen);
						formatted.append(content);
						formatted.append(template.userClose);
						break;
					case "assistant":
						formatted.append(template.assistantOpen);
						formatted.append(content);
						formatted.append(template.assistantClose);
						break;
					default:
						formatted.append(template.userOpen);
						formatted.append(content);
						formatted.append(template.userClose);
						break;
				}
			}
		}

		// Add generation prompt if needed
		if (template.addGenerationPrompt && !messages.isEmpty()) {
			ConversationManager.Message lastMessage = messages.get(messages.size() - 1);
			if ("user".equals(lastMessage.role().toLowerCase())) {
				formatted.append(template.assistantOpen);
			}
		}

		return formatted.toString();
	}

	/**
	 * Auto-detect template type from model name
	 */
	public static TemplateType detectTemplate(String modelName) {
		String lower = modelName.toLowerCase();

		if (lower.contains("llama-3") || lower.contains("llama3")) {
			return TemplateType.LLAMA3;
		} else if (lower.contains("llama-2") || lower.contains("llama2")) {
			return TemplateType.LLAMA2;
		} else if (lower.contains("mistral") || lower.contains("mixtral")) {
			return TemplateType.MISTRAL;
		} else if (lower.contains("vicuna")) {
			return TemplateType.VICUNA;
		} else if (lower.contains("zephyr")) {
			return TemplateType.ZEPHYR;
		} else if (lower.contains("alpaca")) {
			return TemplateType.ALPACA;
		} else if (lower.contains("chatml") || lower.contains("qwen")) {
			return TemplateType.CHATTML;
		}

		// Default to ChatML as it's widely supported
		return TemplateType.CHATTML;
	}

	/**
	 * Parse template from model metadata if available
	 */
	public static Template parseFromMetadata(String chatTemplate) {
		if (chatTemplate == null || chatTemplate.isEmpty()) {
			return TEMPLATES.get(TemplateType.CHATTML);
		}

		// Try to detect known patterns
		if (chatTemplate.contains("<|im_start|>") && chatTemplate.contains("<|im_end|>")) {
			return TEMPLATES.get(TemplateType.CHATTML);
		} else if (chatTemplate.contains("<|start_header_id|>")) {
			return TEMPLATES.get(TemplateType.LLAMA3);
		} else if (chatTemplate.contains("[INST]") && chatTemplate.contains("[/INST]")) {
			if (chatTemplate.contains("<<SYS>>")) {
				return TEMPLATES.get(TemplateType.LLAMA2);
			} else {
				return TEMPLATES.get(TemplateType.MISTRAL);
			}
		}

		// Could parse Jinja2 template here if needed
		return TEMPLATES.get(TemplateType.CHATTML);
	}

	private boolean isFirstUserMessage(List<ConversationManager.Message> messages, ConversationManager.Message current) {
		for (ConversationManager.Message msg : messages) {
			if ("user".equals(msg.role().toLowerCase())) {
				return msg == current;
			}
		}
		return false;
	}

	public TemplateType getType() {
		return type;
	}

	public Template getTemplate() {
		return template;
	}

	/**
	 * Extract raw content from formatted text (reverse formatting)
	 */
	public String extractContent(String formattedText, String role) {
		// Simple extraction - can be enhanced with regex patterns
		String openTag = "";
		String closeTag = "";

		switch (role.toLowerCase()) {
			case "system":
				openTag = template.systemOpen;
				closeTag = template.systemClose;
				break;
			case "user":
				openTag = template.userOpen;
				closeTag = template.userClose;
				break;
			case "assistant":
				openTag = template.assistantOpen;
				closeTag = template.assistantClose;
				break;
		}

		if (!openTag.isEmpty() && formattedText.contains(openTag)) {
			int start = formattedText.indexOf(openTag) + openTag.length();
			int end = closeTag.isEmpty() ? formattedText.length() :
					  formattedText.indexOf(closeTag, start);
			if (end == -1) end = formattedText.length();
			return formattedText.substring(start, end).trim();
		}

		return formattedText;
	}
}
