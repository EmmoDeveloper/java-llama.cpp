package de.kherud.llama;

/**
 * Standard tokens for Llama models.
 * Based on Meta's official documentation for Llama 3.1 and 3.2.
 */
public enum LlamaTokens {
	// Core control tokens
	BEGIN_OF_TEXT("<|begin_of_text|>"),
	END_OF_TEXT("<|end_of_text|>"),

	// Message structure tokens
	START_HEADER_ID("<|start_header_id|>"),
	END_HEADER_ID("<|end_header_id|>"),
	EOT_ID("<|eot_id|>"),  // End of Turn

	// Role identifiers (not tokens, but used with headers)
	SYSTEM_ROLE("system"),
	USER_ROLE("user"),
	ASSISTANT_ROLE("assistant"),
	IPYTHON_ROLE("ipython");  // Tool output role in Llama 3.1+

	private final String token;

	LlamaTokens(String token) {
		this.token = token;
	}

	public String getToken() {
		return token;
	}

	@Override
	public String toString() {
		return token;
	}

	/**
	 * Check if a string contains this token.
	 */
	public boolean isIn(String text) {
		return text != null && text.contains(token);
	}

	/**
	 * Remove this token from a string.
	 */
	public String removeFrom(String text) {
		if (text == null) return null;
		return text.replace(token, "");
	}

	/**
	 * Split text at this token.
	 */
	public String[] split(String text) {
		if (text == null) return new String[0];
		return text.split(java.util.regex.Pattern.quote(token));
	}
}