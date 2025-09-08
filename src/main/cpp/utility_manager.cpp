#include "utility_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "llama_server.h"
#include <llama.h>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>

// These are defined in jllama.cpp but we need access to them
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_utility_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jboolean UtilityManager::supportsGpuOffload(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	return llama_supports_gpu_offload() ? JNI_TRUE : JNI_FALSE;
	
	JNI_CATCH_RET(env, JNI_FALSE)
}

jboolean UtilityManager::supportsMmap(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	return llama_supports_mmap() ? JNI_TRUE : JNI_FALSE;
	
	JNI_CATCH_RET(env, JNI_FALSE)
}

jboolean UtilityManager::supportsMlock(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	return llama_supports_mlock() ? JNI_TRUE : JNI_FALSE;
	
	JNI_CATCH_RET(env, JNI_FALSE)
}

jboolean UtilityManager::supportsRpc(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	return llama_supports_rpc() ? JNI_TRUE : JNI_FALSE;
	
	JNI_CATCH_RET(env, JNI_FALSE)
}

jlong UtilityManager::maxDevices(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	return static_cast<jlong>(llama_max_devices());
	
	JNI_CATCH_RET(env, 0)
}

jlong UtilityManager::maxParallelSequences(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	return static_cast<jlong>(llama_max_parallel_sequences());
	
	JNI_CATCH_RET(env, 0)
}

jstring UtilityManager::printSystemInfo(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	const char* system_info = llama_print_system_info();
	if (!system_info) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get system information");
		return nullptr;
	}
	
	return JniUtils::string_to_jstring(env, system_info);
	
	JNI_CATCH_RET(env, nullptr)
}

jlong UtilityManager::timeUs(JNIEnv* env, jclass cls) {
	JNI_TRY(env)
	
	return static_cast<jlong>(llama_time_us());
	
	JNI_CATCH_RET(env, 0)
}

// Global log callback storage
static JavaVM* g_jvm = nullptr;
static jobject g_log_callback = nullptr;
static std::mutex g_log_mutex;

// C callback function that llama.cpp will call
static void native_log_callback(ggml_log_level level, const char* text, void* user_data) {
	std::lock_guard<std::mutex> lock(g_log_mutex);
	
	if (!g_jvm || !g_log_callback) {
		return;
	}
	
	JNIEnv* env = nullptr;
	bool detach_needed = false;
	
	// Get JNI environment
	int get_env_result = g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
	if (get_env_result == JNI_EDETACHED) {
		if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) == JNI_OK) {
			detach_needed = true;
		} else {
			return;
		}
	} else if (get_env_result != JNI_OK) {
		return;
	}
	
	// Call Java callback
	jclass callback_class = env->GetObjectClass(g_log_callback);
	jmethodID callback_method = env->GetMethodID(callback_class, "onLog", "(ILjava/lang/String;)V");
	
	if (callback_method) {
		jstring j_text = env->NewStringUTF(text);
		env->CallVoidMethod(g_log_callback, callback_method, static_cast<jint>(level), j_text);
		env->DeleteLocalRef(j_text);
	}
	
	env->DeleteLocalRef(callback_class);
	
	if (detach_needed) {
		g_jvm->DetachCurrentThread();
	}
}

void UtilityManager::setLogCallback(JNIEnv* env, jclass cls, jobject callback) {
	JNI_TRY(env)
	
	std::lock_guard<std::mutex> lock(g_log_mutex);
	
	// Clear existing callback
	if (g_log_callback) {
		env->DeleteGlobalRef(g_log_callback);
		g_log_callback = nullptr;
	}
	
	if (callback) {
		// Store Java VM reference
		if (!g_jvm) {
			env->GetJavaVM(&g_jvm);
		}
		
		// Create global reference to callback
		g_log_callback = env->NewGlobalRef(callback);
		
		// Set native callback in llama.cpp
		llama_log_set(native_log_callback, nullptr);
	} else {
		// Clear callback in llama.cpp (default to stderr)
		llama_log_set(nullptr, nullptr);
		g_jvm = nullptr;
	}
	
	JNI_CATCH_RET(env, /* void */)
}

// Global abort callback storage
static std::unordered_map<jlong, jobject> g_abort_callbacks;
static std::mutex g_abort_mutex;

// C callback function for abort operations
static bool native_abort_callback(void* user_data) {
	jlong handle = reinterpret_cast<jlong>(user_data);
	
	std::lock_guard<std::mutex> lock(g_abort_mutex);
	auto it = g_abort_callbacks.find(handle);
	if (it == g_abort_callbacks.end() || !g_jvm) {
		return false;
	}
	
	JNIEnv* env = nullptr;
	bool detach_needed = false;
	
	// Get JNI environment
	int get_env_result = g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
	if (get_env_result == JNI_EDETACHED) {
		if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) == JNI_OK) {
			detach_needed = true;
		} else {
			return false;
		}
	} else if (get_env_result != JNI_OK) {
		return false;
	}
	
	// Call Java callback
	jobject callback = it->second;
	jclass callback_class = env->GetObjectClass(callback);
	jmethodID callback_method = env->GetMethodID(callback_class, "shouldAbort", "()Z");
	
	bool should_abort = false;
	if (callback_method) {
		should_abort = env->CallBooleanMethod(callback, callback_method) == JNI_TRUE;
	}
	
	env->DeleteLocalRef(callback_class);
	
	if (detach_needed) {
		g_jvm->DetachCurrentThread();
	}
	
	return should_abort;
}

void UtilityManager::setAbortCallback(JNIEnv* env, jobject obj, jobject callback) {
	JNI_TRY(env)
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_utility_server(handle);
	if (!server) {
		JNIErrorHandler::throw_illegal_state(env, "Model not loaded");
		return;
	}
	
	std::lock_guard<std::mutex> lock(g_abort_mutex);
	
	// Clear existing callback for this handle
	auto it = g_abort_callbacks.find(handle);
	if (it != g_abort_callbacks.end()) {
		env->DeleteGlobalRef(it->second);
		g_abort_callbacks.erase(it);
	}
	
	if (callback) {
		// Store Java VM reference
		if (!g_jvm) {
			env->GetJavaVM(&g_jvm);
		}
		
		// Create global reference to callback
		jobject global_callback = env->NewGlobalRef(callback);
		g_abort_callbacks[handle] = global_callback;
		
		// Set native callback in llama.cpp
		llama_set_abort_callback(server->ctx, native_abort_callback, reinterpret_cast<void*>(handle));
	} else {
		// Clear callback in llama.cpp
		llama_set_abort_callback(server->ctx, nullptr, nullptr);
	}
	
	JNI_CATCH_RET(env, /* void */)
}