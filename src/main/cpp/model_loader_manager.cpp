#include "model_loader_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "llama_server.h"
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>

// These are defined in jllama.cpp but we need access to them
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_model_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jlong ModelLoaderManager::loadModelFromSplits(JNIEnv* env, jclass cls, jobjectArray paths, jobject params) {
	JNI_TRY(env)
	
	if (!paths) {
		JNIErrorHandler::throw_illegal_argument(env, "Paths array cannot be null");
		return 0;
	}
	
	// Get array length
	jsize pathCount = env->GetArrayLength(paths);
	if (pathCount == 0) {
		JNIErrorHandler::throw_illegal_argument(env, "Paths array cannot be empty");
		return 0;
	}
	
	// Convert Java string array to C++ string vector
	std::vector<std::string> pathStrings;
	std::vector<const char*> pathPtrs;
	pathStrings.reserve(pathCount);
	pathPtrs.reserve(pathCount);
	
	for (jsize i = 0; i < pathCount; i++) {
		jstring jPath = (jstring)env->GetObjectArrayElement(paths, i);
		if (!jPath) {
			JNIErrorHandler::throw_illegal_argument(env, "Path element cannot be null");
			return 0;
		}
		
		std::string pathStr = JniUtils::jstring_to_string(env, jPath);
		pathStrings.push_back(pathStr);
		pathPtrs.push_back(pathStrings.back().c_str());
		
		env->DeleteLocalRef(jPath);
	}
	
	// Create model parameters (using default for now)
	llama_model_params model_params = llama_model_default_params();
	// TODO: Parse Java params object to set model_params if needed
	
	// Initialize backend if not already done
	llama_backend_init();
	
	// Load the model from splits
	llama_model* model = llama_model_load_from_splits(
		pathPtrs.data(),
		pathCount,
		model_params
	);
	
	if (!model) {
		JNIErrorHandler::throw_runtime_exception(env, 
			"Failed to load model from split files");
		return 0;
	}
	
	// Create context with default parameters
	llama_context_params ctx_params = llama_context_default_params();
	llama_context* ctx = llama_new_context_with_model(model, ctx_params);
	
	if (!ctx) {
		llama_free_model(model);
		JNIErrorHandler::throw_runtime_exception(env, 
			"Failed to create context for loaded model");
		return 0;
	}
	
	// Create server instance
	auto server = std::make_unique<LlamaServer>();
	server->model = model;
	server->ctx = ctx;
	
	// Generate unique handle
	jlong handle = reinterpret_cast<jlong>(server.get());
	
	// Store in global map
	{
		std::lock_guard<std::mutex> lock(g_servers_mutex);
		g_servers[handle] = std::move(server);
	}
	
	return handle;
	
	JNI_CATCH_RET(env, 0)
}

void ModelLoaderManager::saveModelToFile(JNIEnv* env, jobject obj, jstring path) {
	JNI_TRY(env)
	
	if (!path) {
		JNIErrorHandler::throw_illegal_argument(env, "Path cannot be null");
		return;
	}
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_model_server(handle);
	if (!server) {
		JNIErrorHandler::throw_illegal_state(env, "Model not loaded");
		return;
	}
	
	std::string pathStr = JniUtils::jstring_to_string(env, path);
	
	// Save the model to file using llama_model_save_to_file
	llama_model_save_to_file(server->model, pathStr.c_str());
	
	JNI_CATCH_RET(env, /* void */)
}