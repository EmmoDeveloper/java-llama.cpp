#include "model_manager.h"
#include "jni_utils.h"
#include "jni_logger.h"
#include "jni_error_handler.h"
#include <mutex>
#include <unordered_map>
#include <memory>

// External global server management (defined in jllama.cpp)
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

void ModelManager::loadModel(JNIEnv* env, jobject obj, jobjectArray args) {
	JNIExceptionGuard exception_guard(env);
	
	// Initialize JNI logger to prevent channel corruption
	JNILogger::initialize(env);
	
	JNI_TRY(env)
	
	// Validate input parameters
	if (!JNIErrorHandler::validate_array(env, args, "args", 2)) {
		return;
	}
	
	// Initialize llama backend
	llama_backend_init();
	
	// Parse model path
	std::string model_path = parseModelPath(env, args);
	if (model_path.empty()) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
			"No model path specified in arguments");
		return;
	}
	
	// Create model parameters with defaults
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = parseGpuLayers(env, args);
	
	// Load model using real llama.cpp API
	llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
	if (!model) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
			"Failed to load model");
		return;
	}
	
	// Create context parameters
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 512; // Default context size for streaming
	
	// Parse additional parameters
	bool embedding_mode = false;
	bool reranking_mode = false;
	parseAdditionalParams(env, args, ctx_params, embedding_mode, reranking_mode);
	
	// Create context
	llama_context* ctx = llama_init_from_model(model, ctx_params);
	if (!ctx) {
		llama_model_free(model);
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
			"Failed to create context");
		return;
	}
	
	// Create sampler
	llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
	llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
	llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
	
	// Create and configure server
	auto server = createServer(model, ctx, sampler, embedding_mode, reranking_mode);
	
	// Start the background server
	server->start_server();
	
	// Store server and return handle
	jlong handle = reinterpret_cast<jlong>(server.get());
	{
		std::lock_guard<std::mutex> lock(g_servers_mutex);
		g_servers[handle] = std::move(server);
	}
	
	// Set the handle in Java object
	jclass cls = env->GetObjectClass(obj);
	jfieldID field = env->GetFieldID(cls, "ctx", "J");
	if (field) {
		env->SetLongField(obj, field, handle);
	}
	
	JNI_CATCH(env)
}

void ModelManager::deleteModel(JNIEnv* env, jobject obj) {
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	
	if (handle != 0) {
		std::lock_guard<std::mutex> lock(g_servers_mutex);
		g_servers.erase(handle);
	}
}

std::string ModelManager::parseModelPath(JNIEnv* env, jobjectArray args) {
	jsize args_length = env->GetArrayLength(args);
	
	// Find the --model argument and get its value
	for (jsize i = 0; i < args_length - 1; i++) {
		jstring arg = (jstring)env->GetObjectArrayElement(args, i);
		std::string arg_str = JniUtils::jstring_to_string(env, arg);
		if (arg_str == "--model") {
			jstring model_path_jstr = (jstring)env->GetObjectArrayElement(args, i + 1);
			return JniUtils::jstring_to_string(env, model_path_jstr);
		}
	}
	return "";
}

int ModelManager::parseGpuLayers(JNIEnv* env, jobjectArray args) {
	jsize args_length = env->GetArrayLength(args);
	
	// Parse GPU layers parameter
	for (jsize i = 0; i < args_length - 1; i++) {
		jstring arg = (jstring)env->GetObjectArrayElement(args, i);
		std::string arg_str = JniUtils::jstring_to_string(env, arg);
		if (arg_str == "--gpu-layers") {
			jstring value_jstr = (jstring)env->GetObjectArrayElement(args, i + 1);
			std::string value_str = JniUtils::jstring_to_string(env, value_jstr);
			return std::stoi(value_str);
		}
	}
	return 0;
}

void ModelManager::parseAdditionalParams(JNIEnv* env, jobjectArray args, 
		llama_context_params& ctx_params, bool& embedding_mode, bool& reranking_mode) {
	jsize args_length = env->GetArrayLength(args);
	
	for (jsize i = 0; i < args_length; i++) {
		jstring arg = (jstring)env->GetObjectArrayElement(args, i);
		std::string arg_str = JniUtils::jstring_to_string(env, arg);
		
		if (arg_str == "--ctx-size" && i + 1 < args_length) {
			jstring value_jstr = (jstring)env->GetObjectArrayElement(args, i + 1);
			std::string value_str = JniUtils::jstring_to_string(env, value_jstr);
			ctx_params.n_ctx = std::stoi(value_str);
		} else if (arg_str == "--threads" && i + 1 < args_length) {
			jstring value_jstr = (jstring)env->GetObjectArrayElement(args, i + 1);
			std::string value_str = JniUtils::jstring_to_string(env, value_jstr);
			ctx_params.n_threads = std::stoi(value_str);
		} else if (arg_str == "--embedding") {
			embedding_mode = true;
			ctx_params.embeddings = true;
		} else if (arg_str == "--reranking") {
			reranking_mode = true;
			ctx_params.embeddings = true; // Reranking requires embeddings to be enabled
		}
	}
}

std::unique_ptr<LlamaServer> ModelManager::createServer(llama_model* model, llama_context* ctx, 
		llama_sampler* sampler, bool embedding_mode, bool reranking_mode) {
	auto server = std::make_unique<LlamaServer>();
	server->model = model;
	server->ctx = ctx;
	server->sampler = sampler;
	server->embedding_mode = embedding_mode;
	server->reranking_mode = reranking_mode;
	return server;
}