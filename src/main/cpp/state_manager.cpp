#include "state_manager.h"
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

jlong StateManager::getStateSize(JNIEnv* env, jobject obj) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);
	
	size_t state_size = llama_state_get_size(server->ctx);
	return (jlong)state_size;
	
	JNI_CATCH_RET(env, -1)
}

jbyteArray StateManager::getStateData(JNIEnv* env, jobject obj) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL(env, server, "server");
	JNI_CHECK_NULL(env, server->ctx, "server->ctx");
	
	size_t state_size = llama_state_get_size(server->ctx);
	// Note: llama.cpp returns 0 on error, but may also return 0 for valid empty states
	// We handle both cases by attempting to create the array regardless
	
	// Create Java byte array
	jbyteArray result = env->NewByteArray(state_size);
	if (!result) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to allocate byte array for state data");
		return nullptr;
	}
	
	// Get array elements for writing
	jbyte* state_bytes = env->GetByteArrayElements(result, nullptr);
	if (!state_bytes) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get byte array elements");
		return nullptr;
	}
	
	// Copy state data
	size_t copied_bytes = llama_state_get_data(server->ctx, (uint8_t*)state_bytes, state_size);
	
	// Release array
	env->ReleaseByteArrayElements(result, state_bytes, 0);
	
	if (copied_bytes != state_size) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to copy complete state data");
		return nullptr;
	}
	
	return result;
	
	JNI_CATCH_RET(env, nullptr)
}

jlong StateManager::setStateData(JNIEnv* env, jobject obj, jbyteArray state_data) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_array(env, state_data, "state_data", 1)) {
		return -1;
	}
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);
	
	jsize data_size = env->GetArrayLength(state_data);
	jbyte* state_bytes = env->GetByteArrayElements(state_data, nullptr);
	if (!state_bytes) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get state data bytes");
		return -1;
	}
	
	// Set state data
	size_t loaded_bytes = llama_state_set_data(server->ctx, (const uint8_t*)state_bytes, data_size);
	
	// Release array
	env->ReleaseByteArrayElements(state_data, state_bytes, JNI_ABORT);
	
	return (jlong)loaded_bytes;
	
	JNI_CATCH_RET(env, -1)
}

jboolean StateManager::saveStateToFile(JNIEnv* env, jobject obj, jstring path, jintArray tokens) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_string(env, path, "path")) {
		return JNI_FALSE;
	}
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", JNI_FALSE);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", JNI_FALSE);
	
	std::string file_path = JniUtils::jstring_to_string(env, path);
	
	// Handle tokens array (can be null for no tokens)
	llama_token* token_data = nullptr;
	jsize token_count = 0;
	jint* token_elements = nullptr;
	
	if (tokens) {
		token_count = env->GetArrayLength(tokens);
		if (token_count > 0) {
			token_elements = env->GetIntArrayElements(tokens, nullptr);
			if (!token_elements) {
				JNIErrorHandler::throw_runtime_exception(env, "Failed to get token array elements");
				return JNI_FALSE;
			}
			token_data = (llama_token*)token_elements;
		}
	}
	
	// Save state to file
	bool success = llama_state_save_file(server->ctx, file_path.c_str(), token_data, token_count);
	
	// Release token array if used
	if (token_elements) {
		env->ReleaseIntArrayElements(tokens, token_elements, JNI_ABORT);
	}
	
	return success ? JNI_TRUE : JNI_FALSE;
	
	JNI_CATCH_RET(env, JNI_FALSE)
}

jintArray StateManager::loadStateFromFile(JNIEnv* env, jobject obj, jstring path, jint max_tokens) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_string(env, path, "path")) {
		return nullptr;
	}
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL(env, server, "server");
	JNI_CHECK_NULL(env, server->ctx, "server->ctx");
	
	std::string file_path = JniUtils::jstring_to_string(env, path);
	
	// Create buffer for tokens
	std::vector<llama_token> tokens(max_tokens > 0 ? max_tokens : 4096);
	size_t token_count = 0;
	
	// Load state from file
	bool success = llama_state_load_file(server->ctx, file_path.c_str(), 
		tokens.data(), tokens.size(), &token_count);
	
	if (!success) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to load state from file");
		return nullptr;
	}
	
	// Create Java int array for tokens
	jintArray result = env->NewIntArray(token_count);
	if (!result) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to allocate token array");
		return nullptr;
	}
	
	if (token_count > 0) {
		env->SetIntArrayRegion(result, 0, token_count, (jint*)tokens.data());
	}
	
	return result;
	
	JNI_CATCH_RET(env, nullptr)
}

jlong StateManager::getSequenceStateSize(JNIEnv* env, jobject obj, jint seq_id) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);
	
	size_t state_size = llama_state_seq_get_size(server->ctx, seq_id);
	// If state size is 0, it might mean the sequence is empty or doesn't exist
	// This is not necessarily an error in llama.cpp context
	return (jlong)state_size;
	
	JNI_CATCH_RET(env, -1)
}

jbyteArray StateManager::getSequenceStateData(JNIEnv* env, jobject obj, jint seq_id) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL(env, server, "server");
	JNI_CHECK_NULL(env, server->ctx, "server->ctx");
	
	size_t state_size = llama_state_seq_get_size(server->ctx, seq_id);
	if (state_size == 0) {
		// Return empty byte array instead of throwing exception
		// This matches the behavior expected by the Java wrapper
		return env->NewByteArray(0);
	}
	
	// Create Java byte array
	jbyteArray result = env->NewByteArray(state_size);
	if (!result) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to allocate byte array for sequence state");
		return nullptr;
	}
	
	// Get array elements
	jbyte* state_bytes = env->GetByteArrayElements(result, nullptr);
	if (!state_bytes) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get byte array elements");
		return nullptr;
	}
	
	// Copy sequence state data
	size_t copied_bytes = llama_state_seq_get_data(server->ctx, (uint8_t*)state_bytes, state_size, seq_id);
	
	// Release array
	env->ReleaseByteArrayElements(result, state_bytes, 0);
	
	if (copied_bytes != state_size) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to copy complete sequence state");
		return nullptr;
	}
	
	return result;
	
	JNI_CATCH_RET(env, nullptr)
}

jlong StateManager::setSequenceStateData(JNIEnv* env, jobject obj, jbyteArray state_data, jint seq_id) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_array(env, state_data, "state_data", 1)) {
		return -1;
	}
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);
	
	jsize data_size = env->GetArrayLength(state_data);
	jbyte* state_bytes = env->GetByteArrayElements(state_data, nullptr);
	if (!state_bytes) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get state data bytes");
		return -1;
	}
	
	// Set sequence state data
	size_t loaded_bytes = llama_state_seq_set_data(server->ctx, (const uint8_t*)state_bytes, data_size, seq_id);
	
	// Release array
	env->ReleaseByteArrayElements(state_data, state_bytes, JNI_ABORT);
	
	return (jlong)loaded_bytes;
	
	JNI_CATCH_RET(env, -1)
}

jlong StateManager::saveSequenceToFile(JNIEnv* env, jobject obj, jstring path, jint seq_id, jintArray tokens) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_string(env, path, "path")) {
		return -1;
	}
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);
	
	std::string file_path = JniUtils::jstring_to_string(env, path);
	
	// Handle tokens array (can be null)
	llama_token* token_data = nullptr;
	jsize token_count = 0;
	jint* token_elements = nullptr;
	
	if (tokens) {
		token_count = env->GetArrayLength(tokens);
		if (token_count > 0) {
			token_elements = env->GetIntArrayElements(tokens, nullptr);
			if (!token_elements) {
				JNIErrorHandler::throw_runtime_exception(env, "Failed to get token array elements");
				return -1;
			}
			token_data = (llama_token*)token_elements;
		}
	}
	
	// Save sequence to file
	size_t saved_bytes = llama_state_seq_save_file(server->ctx, file_path.c_str(), seq_id, token_data, token_count);
	
	// Release token array if used
	if (token_elements) {
		env->ReleaseIntArrayElements(tokens, token_elements, JNI_ABORT);
	}
	
	return (jlong)saved_bytes;
	
	JNI_CATCH_RET(env, -1)
}

jintArray StateManager::loadSequenceFromFile(JNIEnv* env, jobject obj, jstring path, jint seq_id, jint max_tokens) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_string(env, path, "path")) {
		return nullptr;
	}
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL(env, server, "server");
	JNI_CHECK_NULL(env, server->ctx, "server->ctx");
	
	std::string file_path = JniUtils::jstring_to_string(env, path);
	
	// Create buffer for tokens
	std::vector<llama_token> tokens(max_tokens > 0 ? max_tokens : 4096);
	size_t token_count = 0;
	
	// Load sequence from file
	size_t loaded_bytes = llama_state_seq_load_file(server->ctx, file_path.c_str(), seq_id,
		tokens.data(), tokens.size(), &token_count);
	
	if (loaded_bytes == 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to load sequence from file");
		return nullptr;
	}
	
	// Create Java int array for tokens
	jintArray result = env->NewIntArray(token_count);
	if (!result) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to allocate token array");
		return nullptr;
	}
	
	if (token_count > 0) {
		env->SetIntArrayRegion(result, 0, token_count, (jint*)tokens.data());
	}
	
	return result;
	
	JNI_CATCH_RET(env, nullptr)
}

LlamaServer* StateManager::getServer(JNIEnv* env, jobject obj) {
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