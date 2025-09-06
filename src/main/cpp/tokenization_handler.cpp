#include "tokenization_handler.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include <mutex>
#include <unordered_map>
#include <memory>

// External global server management (defined in jllama.cpp)
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jintArray TokenizationHandler::encode(JNIEnv* env, jobject obj, jstring text) {
	// Validate input parameters first 
	if (!JNIErrorHandler::validate_string(env, text, "text")) {
		// Double-check that exception is actually pending
		if (!env->ExceptionCheck()) {
			// If no exception pending, explicitly throw one
			env->ThrowNew(env->FindClass("java/lang/NullPointerException"), 
				"text string parameter is null");
		}
		return nullptr; // Exception is already set, just return immediately
	}
	
	JNI_TRY(env)
	
	// Get server handle
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL(env, server, "server");
	
	std::string input = JniUtils::jstring_to_string(env, text);
	
	// Tokenize using real llama.cpp API
	const llama_vocab* vocab = llama_model_get_vocab(server->model);
	std::vector<llama_token> tokens;
	
	int n_tokens = tokenizeText(vocab, input, tokens);
	if (n_tokens < 0) {
		return nullptr;
	}
	
	// Convert to Java int array
	jintArray result = env->NewIntArray(n_tokens);
	if (result) {
		env->SetIntArrayRegion(result, 0, n_tokens, (jint*)tokens.data());
	}
	
	return result;
	
	JNI_CATCH_RET(env, nullptr)
}

jbyteArray TokenizationHandler::decodeBytes(JNIEnv* env, jobject obj, jintArray token_array) {
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_server(handle);
	if (!server) return nullptr;
	
	// Get tokens from Java array
	jsize len = env->GetArrayLength(token_array);
	jint* tokens = env->GetIntArrayElements(token_array, nullptr);
	
	// Detokenize using real llama.cpp API - process all tokens at once
	const llama_vocab* vocab = llama_model_get_vocab(server->model);
	std::string result = detokenizeTokens(vocab, (llama_token*)tokens, len);
	
	env->ReleaseIntArrayElements(token_array, tokens, JNI_ABORT);
	
	// Convert to Java byte array
	jbyteArray byte_array = env->NewByteArray(result.length());
	if (byte_array) {
		env->SetByteArrayRegion(byte_array, 0, result.length(), 
			(jbyte*)result.data());
	}
	
	return byte_array;
}

LlamaServer* TokenizationHandler::getServer(JNIEnv* env, jobject obj) {
	jclass cls = env->GetObjectClass(obj);
	if (!cls) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get object class");
		return nullptr;
	}
	
	jfieldID field = env->GetFieldID(cls, "ctx", "J");
	if (!field) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get ctx field");
		return nullptr;
	}
	
	jlong handle = env->GetLongField(obj, field);
	return get_server(handle);
}

int TokenizationHandler::tokenizeText(const llama_vocab* vocab, const std::string& text, 
		std::vector<llama_token>& tokens) {
	tokens.resize(text.length() + 1);
	
	int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), 
		tokens.data(), tokens.size(), true, false);
	
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		n_tokens = llama_tokenize(vocab, text.c_str(), text.length(),
			tokens.data(), tokens.size(), true, false);
	}
	
	if (n_tokens > 0) {
		tokens.resize(n_tokens);
	}
	
	return n_tokens;
}

std::string TokenizationHandler::detokenizeTokens(const llama_vocab* vocab, 
		const llama_token* tokens, int num_tokens) {
	// Allocate buffer for the decoded text
	size_t max_len = num_tokens * 32; // Assume max 32 bytes per token
	std::vector<char> buffer(max_len);
	
	// Decode all tokens at once to preserve spaces properly
	int result_len = llama_detokenize(vocab, tokens, num_tokens, 
		buffer.data(), buffer.size(), false, false);
	
	std::string result;
	if (result_len > 0) {
		result.assign(buffer.data(), result_len);
	}
	
	return result;
}