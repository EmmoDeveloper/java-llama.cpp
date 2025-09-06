#include "lora_adapter_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include <mutex>
#include <unordered_map>
#include <memory>
#include <algorithm>

// External global server management (defined in jllama.cpp)
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jlong LoRAAdapterManager::loadAdapter(JNIEnv* env, jobject obj, jstring path_lora) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_string(env, path_lora, "path_lora")) {
		return -1;
	}

	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->model, "server->model", -1);

	std::string lora_path = JniUtils::jstring_to_string(env, path_lora);
	
	// Load LoRA adapter
	llama_adapter_lora* adapter = llama_adapter_lora_init(server->model, lora_path.c_str());
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to load LoRA adapter from: " + lora_path);
		return -1;
	}

	// Return adapter pointer as handle
	return reinterpret_cast<jlong>(adapter);

	JNI_CATCH_RET(env, -1)
}

void LoRAAdapterManager::freeAdapter(JNIEnv* env, jlong adapter_handle) {
	JNI_TRY(env)
	
	// Early validation before calling getAdapter
	if (adapter_handle <= 0 || adapter_handle == -1 || 
		adapter_handle < 0x1000 || adapter_handle >= 0x7FFFFFFFFFFFFFFFLL) {
		// Invalid handle, silently return without attempting to free
		return;
	}
	
	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (adapter) {
		llama_adapter_lora_free(adapter);
	}

	JNI_CATCH(env)
}

jint LoRAAdapterManager::setAdapter(JNIEnv* env, jobject obj, jlong adapter_handle, jfloat scale) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);

	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return -1;
	}

	// Apply LoRA adapter to context
	int32_t result = llama_set_adapter_lora(server->ctx, adapter, scale);
	return (jint)result;

	JNI_CATCH_RET(env, -1)
}

jint LoRAAdapterManager::removeAdapter(JNIEnv* env, jobject obj, jlong adapter_handle) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);

	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return -1;
	}

	// Remove LoRA adapter from context
	int32_t result = llama_rm_adapter_lora(server->ctx, adapter);
	return (jint)result;

	JNI_CATCH_RET(env, -1)
}

void LoRAAdapterManager::clearAdapters(JNIEnv* env, jobject obj) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_VOID(env, server, "server");
	JNI_CHECK_NULL_VOID(env, server->ctx, "server->ctx");

	// Clear all LoRA adapters from context
	llama_clear_adapter_lora(server->ctx);

	JNI_CATCH(env)
}

jint LoRAAdapterManager::applyControlVector(JNIEnv* env, jobject obj, jfloatArray data) {
	JNI_TRY(env)
	
	LlamaServer* server = getServer(env, obj);
	JNI_CHECK_NULL_RET(env, server, "server", -1);
	JNI_CHECK_NULL_RET(env, server->ctx, "server->ctx", -1);

	if (!data) {
		// Clear control vector by passing null data
		int32_t result = llama_apply_adapter_cvec(server->ctx, nullptr, 0, 0, 0, -1);
		return (jint)result;
	}

	if (!JNIErrorHandler::validate_array(env, data, "data", 1)) {
		return -1;
	}

	jsize data_length = env->GetArrayLength(data);
	jfloat* data_elements = env->GetFloatArrayElements(data, nullptr);
	if (!data_elements) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get control vector data");
		return -1;
	}

	// Apply control vector across all layers
	int n_embd = llama_model_n_embd(server->model);
	int n_layers = llama_model_n_layer(server->model);
	// Calculate how many layers the data can cover (data_length / n_embd)
	int layers_available = data_length / n_embd;
	int il_end = std::min(layers_available - 1, n_layers - 1);
	int32_t result = llama_apply_adapter_cvec(server->ctx, data_elements, data_length, n_embd, 0, il_end);

	// Release array
	env->ReleaseFloatArrayElements(data, data_elements, JNI_ABORT);

	return (jint)result;

	JNI_CATCH_RET(env, -1)
}

jstring LoRAAdapterManager::getAdapterMetaValue(JNIEnv* env, jlong adapter_handle, jstring key) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_string(env, key, "key")) {
		return nullptr;
	}

	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return nullptr;
	}

	std::string key_str = JniUtils::jstring_to_string(env, key);
	
	// Get buffer size needed
	int32_t buf_size = llama_adapter_meta_val_str(adapter, key_str.c_str(), nullptr, 0);
	if (buf_size <= 0) {
		return nullptr; // Key not found or error
	}

	// Allocate buffer and get value
	std::vector<char> buffer(buf_size);
	int32_t actual_size = llama_adapter_meta_val_str(adapter, key_str.c_str(), buffer.data(), buf_size);
	if (actual_size <= 0) {
		return nullptr;
	}

	return JniUtils::string_to_jstring(env, std::string(buffer.data(), actual_size));

	JNI_CATCH_RET(env, nullptr)
}

jint LoRAAdapterManager::getAdapterMetaCount(JNIEnv* env, jlong adapter_handle) {
	JNI_TRY(env)
	
	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return -1;
	}

	int32_t count = llama_adapter_meta_count(adapter);
	return (jint)count;

	JNI_CATCH_RET(env, -1)
}

jstring LoRAAdapterManager::getAdapterMetaKeyByIndex(JNIEnv* env, jlong adapter_handle, jint index) {
	JNI_TRY(env)
	
	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return nullptr;
	}

	// Get buffer size needed
	int32_t buf_size = llama_adapter_meta_key_by_index(adapter, index, nullptr, 0);
	if (buf_size <= 0) {
		return nullptr; // Index out of bounds or error
	}

	// Allocate buffer and get key
	std::vector<char> buffer(buf_size);
	int32_t actual_size = llama_adapter_meta_key_by_index(adapter, index, buffer.data(), buf_size);
	if (actual_size <= 0) {
		return nullptr;
	}

	return JniUtils::string_to_jstring(env, std::string(buffer.data(), actual_size));

	JNI_CATCH_RET(env, nullptr)
}

jstring LoRAAdapterManager::getAdapterMetaValueByIndex(JNIEnv* env, jlong adapter_handle, jint index) {
	JNI_TRY(env)
	
	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return nullptr;
	}

	// Get buffer size needed
	int32_t buf_size = llama_adapter_meta_val_str_by_index(adapter, index, nullptr, 0);
	if (buf_size <= 0) {
		return nullptr; // Index out of bounds or error
	}

	// Allocate buffer and get value
	std::vector<char> buffer(buf_size);
	int32_t actual_size = llama_adapter_meta_val_str_by_index(adapter, index, buffer.data(), buf_size);
	if (actual_size <= 0) {
		return nullptr;
	}

	return JniUtils::string_to_jstring(env, std::string(buffer.data(), actual_size));

	JNI_CATCH_RET(env, nullptr)
}

jlong LoRAAdapterManager::getAloraInvocationTokenCount(JNIEnv* env, jlong adapter_handle) {
	JNI_TRY(env)
	
	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return 0;
	}

	uint64_t count = llama_adapter_get_alora_n_invocation_tokens(adapter);
	return (jlong)count;

	JNI_CATCH_RET(env, 0)
}

jintArray LoRAAdapterManager::getAloraInvocationTokens(JNIEnv* env, jlong adapter_handle) {
	JNI_TRY(env)
	
	llama_adapter_lora* adapter = getAdapter(adapter_handle);
	if (!adapter) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid adapter handle");
		return nullptr;
	}

	uint64_t token_count = llama_adapter_get_alora_n_invocation_tokens(adapter);
	if (token_count == 0) {
		return env->NewIntArray(0); // Return empty array for non-ALORA adapters
	}

	const llama_token* tokens = llama_adapter_get_alora_invocation_tokens(adapter);
	if (!tokens) {
		return env->NewIntArray(0);
	}

	// Create Java int array
	jintArray result = env->NewIntArray(token_count);
	if (!result) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to allocate token array");
		return nullptr;
	}

	env->SetIntArrayRegion(result, 0, token_count, (jint*)tokens);
	return result;

	JNI_CATCH_RET(env, nullptr)
}

LlamaServer* LoRAAdapterManager::getServer(JNIEnv* env, jobject obj) {
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

llama_adapter_lora* LoRAAdapterManager::getAdapter(jlong handle) {
	// Reject invalid handles immediately
	if (handle <= 0 || handle == -1) {
		return nullptr;
	}
	
	// Additional safety check - reject handles that are clearly invalid pointers
	// Real pointers should be in reasonable memory ranges
	// Reject Long.MAX_VALUE and other extreme values
	if (handle < 0x1000 || handle >= 0x7FFFFFFFFFFFFFFFLL) {
		return nullptr;
	}
	
	return reinterpret_cast<llama_adapter_lora*>(handle);
}