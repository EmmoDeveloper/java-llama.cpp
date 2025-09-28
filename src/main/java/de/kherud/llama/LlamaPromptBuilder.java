package de.kherud.llama;

import java.util.ArrayList;
import java.util.List;

import static de.kherud.llama.LlamaTokens.ASSISTANT_ROLE;
import static de.kherud.llama.LlamaTokens.BEGIN_OF_TEXT;
import static de.kherud.llama.LlamaTokens.END_HEADER_ID;
import static de.kherud.llama.LlamaTokens.EOT_ID;
import static de.kherud.llama.LlamaTokens.IPYTHON_ROLE;
import static de.kherud.llama.LlamaTokens.START_HEADER_ID;
import static de.kherud.llama.LlamaTokens.SYSTEM_ROLE;
import static de.kherud.llama.LlamaTokens.USER_ROLE;

/**
 * Utility class for building and parsing Llama 3.1+ instruction prompts.
 * Handles proper token formatting and message structure.
 */
public class LlamaPromptBuilder {

	private String systemMessage;
	private final List<Message> messages = new ArrayList<>();
	private boolean includeTimestamps = false;

	public static class Message {
		private final String role;
		private final String content;

		public Message(String role, String content) {
			this.role = role;
			this.content = content;
		}

		public String getRole() {
			return role;
		}

		public String getContent() {
			return content;
		}
	}

	/**
	 * Set the system message for the conversation.
	 */
	public LlamaPromptBuilder withSystem(String message) {
		this.systemMessage = message;
		return this;
	}

	/**
	 * Add a user message.
	 */
	public LlamaPromptBuilder addUser(String message) {
		messages.add(new Message(USER_ROLE.getToken(), message));
		return this;
	}

	/**
	 * Add an assistant message.
	 */
	public LlamaPromptBuilder addAssistant(String message) {
		messages.add(new Message(ASSISTANT_ROLE.getToken(), message));
		return this;
	}

	/**
	 * Add a tool output message (Llama 3.1+ only).
	 */
	public LlamaPromptBuilder addToolOutput(String output) {
		messages.add(new Message(IPYTHON_ROLE.getToken(), output));
		return this;
	}

	/**
	 * Include default timestamps in system message.
	 */
	public LlamaPromptBuilder includeTimestamps(boolean include) {
		this.includeTimestamps = include;
		return this;
	}

	/**
	 * Build the complete prompt with proper token formatting.
	 */
	public String build() {
		StringBuilder prompt = new StringBuilder();
		prompt.append(BEGIN_OF_TEXT.getToken());

		// Add system message if present
		if (systemMessage != null || includeTimestamps) {
			prompt.append(START_HEADER_ID.getToken())
				.append(SYSTEM_ROLE.getToken())
				.append(END_HEADER_ID.getToken())
				.append("\n\n");

			if (includeTimestamps) {
				prompt.append("Cutting Knowledge Date: December 2023\n");
				prompt.append("Today Date: ").append(java.time.LocalDate.now()).append("\n\n");
			}

			if (systemMessage != null) {
				prompt.append(systemMessage);
			}

			prompt.append(EOT_ID.getToken());
		}

		// Add conversation messages
		for (Message msg : messages) {
			prompt.append(START_HEADER_ID.getToken())
				.append(msg.role)
				.append(END_HEADER_ID.getToken())
				.append("\n\n")
				.append(msg.content)
				.append(EOT_ID.getToken());
		}

		// End with assistant header to prompt generation
		prompt.append(START_HEADER_ID.getToken())
			.append(ASSISTANT_ROLE.getToken())
			.append(END_HEADER_ID.getToken())
			.append("\n\n");

		return prompt.toString();
	}

	/**
	 * Create a simple chat prompt for a single user message.
	 */
	public static String simpleChat(String userMessage) {
		return new LlamaPromptBuilder()
			.addUser(userMessage)
			.build();
	}

	/**
	 * Parse and clean the model's response, removing any special tokens.
	 */
	public static String parseResponse(String modelOutput) {
		if (modelOutput == null) return "";

		String result = modelOutput;

		// Remove any trailing special tokens
		for (LlamaTokens token : LlamaTokens.values()) {
			if (token.isIn(result)) {
				// For EOT_ID, split and take only the content before it
				if (token == EOT_ID) {
					String[] parts = token.split(result);
					if (parts.length > 0) {
						result = parts[0];
					}
				} else {
					// Remove other tokens entirely
					result = token.removeFrom(result);
				}
			}
		}

		return result.trim();
	}

	/**
	 * Extract messages from a raw prompt string.
	 */
	public static List<Message> parsePrompt(String prompt) {
		List<Message> messages = new ArrayList<>();

		if (prompt == null || prompt.isEmpty()) {
			return messages;
		}

		// Split by header start tokens
		String[] parts = START_HEADER_ID.split(prompt);

		for (String part : parts) {
			if (part.contains(END_HEADER_ID.getToken())) {
				String[] headerAndContent = END_HEADER_ID.split(part);
				if (headerAndContent.length >= 2) {
					String role = headerAndContent[0].trim();
					String contentWithEot = headerAndContent[1];

					// Remove EOT token if present
					String content = EOT_ID.isIn(contentWithEot)
						? EOT_ID.split(contentWithEot)[0]
						: contentWithEot;

					// Clean up the content
					content = content.replaceFirst("^\\n\\n", "").trim();

					if (!content.isEmpty()) {
						messages.add(new Message(role, content));
					}
				}
			}
		}

		return messages;
	}
}
