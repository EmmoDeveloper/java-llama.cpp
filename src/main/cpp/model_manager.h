#pragma once

#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"
#include "llama_server.h"
#include "jni_error_handler.h"

/**
 * Handles model loading, initialization, and cleanup operations.
 * Manages the lifecycle of llama.cpp models and contexts.
 */
class ModelManager {
public:
	/**
	 * Load a model from parameters array and initialize the context.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object 
	 * @param args Model parameters array
	 */
	static void loadModel(JNIEnv* env, jobject obj, jobjectArray args);

	/**
	 * Clean up model and context resources.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 */
	static void deleteModel(JNIEnv* env, jobject obj);

private:
	/**
	 * Parse model path from arguments array.
	 * @param env JNI environment
	 * @param args Arguments array
	 * @return Model path string, empty if not found
	 */
	static std::string parseModelPath(JNIEnv* env, jobjectArray args);

	/**
	 * Parse GPU layers parameter from arguments.
	 * @param env JNI environment
	 * @param args Arguments array
	 * @return Number of GPU layers, 0 if not specified
	 */
	static int parseGpuLayers(JNIEnv* env, jobjectArray args);

	/**
	 * Parse additional model parameters from arguments.
	 * @param env JNI environment
	 * @param args Arguments array
	 * @param ctx_params Context parameters to populate
	 * @param embedding_mode Output parameter for embedding mode
	 * @param reranking_mode Output parameter for reranking mode
	 */
	static void parseAdditionalParams(JNIEnv* env, jobjectArray args, 
		llama_context_params& ctx_params, bool& embedding_mode, bool& reranking_mode);

	/**
	 * Create and configure a LlamaServer instance.
	 * @param model Loaded llama model
	 * @param ctx Initialized llama context
	 * @param sampler Configured sampler
	 * @param embedding_mode Whether embedding mode is enabled
	 * @param reranking_mode Whether reranking mode is enabled
	 * @return Unique pointer to configured server
	 */
	static std::unique_ptr<LlamaServer> createServer(llama_model* model, llama_context* ctx, 
		llama_sampler* sampler, bool embedding_mode, bool reranking_mode);
};