#include "embedding_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "llama_server.h"
#include "memory_manager.h"
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>

// These are defined in jllama.cpp but we need access to them
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_embedding_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jfloatArray EmbeddingManager::createEmbedding(JNIEnv* env, jobject obj, jstring text) {
	JNI_TRY(env)

	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_embedding_server(handle);
	if (!server) return nullptr;
	
	// Check if embedding mode is enabled
	if (!server->embedding_mode) {
		JNIErrorHandler::throw_illegal_state(env, 
			"Model was not loaded with embedding support (see ModelParameters#enableEmbedding())");
		return nullptr;
	}
	
	// Original code (commented out for testing)
	/*
	if (!server->embedding_mode) {
		JNI_LOG_ERROR("DEBUG: Throwing IllegalStateException because embedding_mode is false");
		JNIErrorHandler::throw_illegal_state(env, 
			"Model was not loaded with embedding support (see ModelParameters#enableEmbedding())");
		JNI_LOG_ERROR("DEBUG: Exception thrown, returning nullptr");
		return nullptr;
	}
	*/
	
	std::string input = JniUtils::jstring_to_string(env, text);
	
	// Tokenize the input text
	const llama_vocab* vocab = llama_model_get_vocab(server->model);
	std::vector<llama_token> tokens;
	tokens.resize(input.length() + 1);
	
	int n_tokens = llama_tokenize(vocab, input.c_str(), input.length(), 
								  tokens.data(), tokens.size(), true, false);
	
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		n_tokens = llama_tokenize(vocab, input.c_str(), input.length(),
								  tokens.data(), tokens.size(), true, false);
	}
	
	if (n_tokens < 0) {
		JNIErrorHandler::throw_runtime_exception(env, 
			"Failed to tokenize input for embedding");
		return nullptr;
	}
	
	tokens.resize(n_tokens);
	
	// Clear previous memory (embeddings don't need persistent context)
	llama_memory_clear(llama_get_memory(server->ctx), true);
	
	// Use RAII for batch management to ensure cleanup
	BatchRAII batch_raii(n_tokens, 0, 1);
	llama_batch* batch = batch_raii.get();
	
	for (int i = 0; i < n_tokens; i++) {
		batch->token[i] = tokens[i];
		batch->pos[i] = i;
		batch->n_seq_id[i] = 1;
		batch->seq_id[i][0] = 0;
		batch->logits[i] = true; // We need embeddings for all tokens or just the last one
	}
	batch->n_tokens = n_tokens;
	
	// Process the batch to compute embeddings
	if (llama_decode(server->ctx, *batch) != 0) {
		JNIErrorHandler::throw_runtime_exception(env, 
			"Failed to compute embeddings");
		return nullptr;
	}
	
	// Get embedding dimension
	int n_embd = llama_model_n_embd(server->model);
	
	// Get embeddings based on pooling type
	const float* embd = nullptr;
	enum llama_pooling_type pooling_type = llama_pooling_type(server->ctx);
	
	if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
		// For models without pooling, get embedding from the last token
		embd = llama_get_embeddings_ith(server->ctx, n_tokens - 1);
	} else {
		// For models with pooling, get the sequence embedding
		embd = llama_get_embeddings_seq(server->ctx, 0);
	}
	
	// Batch will be automatically freed by RAII
	
	if (!embd) {
		JNIErrorHandler::throw_runtime_exception(env, 
			"Failed to get embeddings from context");
		return nullptr;
	}
	
	// Create Java float array and copy embeddings
	jfloatArray result = env->NewFloatArray(n_embd);
	if (!result) {
		JNIErrorHandler::throw_out_of_memory(env, 
			"Could not allocate embedding array");
		return nullptr;
	}
	
	env->SetFloatArrayRegion(result, 0, n_embd, embd);
	
	return result;

	JNI_CATCH_RET(env, nullptr)
}

jfloatArray EmbeddingManager::getAllEmbeddings(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_embedding_server(handle);
	if (!server) return nullptr;
	
	// Get all embeddings from the context using llama_get_embeddings
	const float* embd = llama_get_embeddings(server->ctx);
	if (!embd) {
		JNIErrorHandler::throw_runtime_exception(env, 
			"No embeddings available - ensure context has been processed and embeddings are enabled");
		return nullptr;
	}
	
	// Get embedding dimension
	int n_embd = llama_model_n_embd(server->model);
	
	// Create Java float array and copy embeddings
	jfloatArray result = env->NewFloatArray(n_embd);
	if (!result) {
		JNIErrorHandler::throw_out_of_memory(env, 
			"Could not allocate embedding array");
		return nullptr;
	}
	
	env->SetFloatArrayRegion(result, 0, n_embd, embd);
	
	return result;

	JNI_CATCH_RET(env, nullptr)
}

void EmbeddingManager::setEmbeddingMode(JNIEnv* env, jobject obj, jboolean embeddings) {
	JNI_TRY(env)

	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_embedding_server(handle);
	if (!server) return;
	
	// Set embedding mode using llama_set_embeddings
	llama_set_embeddings(server->ctx, embeddings == JNI_TRUE);

	JNI_CATCH_RET(env, /* void */)
}