#pragma once

#include <jni.h>
#include "llama.h"
#include "llama_server.h"

/**
 * Handles text tokenization and detokenization operations.
 * Provides encoding/decoding functionality for text and tokens.
 */
class TokenizationHandler {
public:
	/**
	 * Encode text to token array.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param text Text to encode
	 * @return Java int array of tokens, or nullptr on error
	 */
	static jintArray encode(JNIEnv* env, jobject obj, jstring text);

	/**
	 * Decode token array to byte array.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param token_array Array of tokens to decode
	 * @return Java byte array containing decoded text, or nullptr on error
	 */
	static jbyteArray decodeBytes(JNIEnv* env, jobject obj, jintArray token_array);

private:
	/**
	 * Get server handle from Java object.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @return LlamaServer pointer, or nullptr if not found
	 */
	static LlamaServer* getServer(JNIEnv* env, jobject obj);

	/**
	 * Tokenize text using llama.cpp vocabulary.
	 * @param vocab Vocabulary to use
	 * @param text Text to tokenize
	 * @param tokens Output vector for tokens
	 * @return Number of tokens, negative on error
	 */
	static int tokenizeText(const llama_vocab* vocab, const std::string& text, 
		std::vector<llama_token>& tokens);

	/**
	 * Detokenize tokens using llama.cpp vocabulary.
	 * @param vocab Vocabulary to use
	 * @param tokens Array of tokens
	 * @param num_tokens Number of tokens
	 * @return Decoded text string
	 */
	static std::string detokenizeTokens(const llama_vocab* vocab, 
		const llama_token* tokens, int num_tokens);
};