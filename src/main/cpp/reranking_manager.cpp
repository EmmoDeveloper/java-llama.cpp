#include "reranking_manager.h"
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

static LlamaServer* get_reranking_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jobject RerankingManager::rerank(JNIEnv* env, jobject obj, jstring query, jobjectArray documents) {
	JNI_TRY(env)
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_reranking_server(handle);
	if (!server) return nullptr;
	
	// Check if reranking mode is enabled
	if (!server->reranking_mode) {
		JNIErrorHandler::throw_illegal_state(env,
			"Model was not loaded with reranking support (see ModelParameters#enableReranking())");
		return nullptr;
	}
	
	std::string query_str = JniUtils::jstring_to_string(env, query);
	
	// Get documents from Java array
	jsize num_documents = env->GetArrayLength(documents);
	if (num_documents == 0) {
		JNIErrorHandler::throw_illegal_argument(env,
			"No documents provided for reranking");
		return nullptr;
	}
	
	const llama_vocab* vocab = llama_model_get_vocab(server->model);
	
	// Tokenize query
	std::vector<llama_token> query_tokens = tokenizeText(vocab, query_str);
	if (query_tokens.empty()) {
		JNIErrorHandler::throw_runtime_exception(env,
			"Failed to tokenize query for reranking");
		return nullptr;
	}
	
	// Create LlamaOutput result object
	jclass output_class = env->FindClass("de/kherud/llama/LlamaOutput");
	if (!output_class) return nullptr;
	
	jmethodID constructor = env->GetMethodID(output_class, "<init>", "([BLjava/util/Map;Z)V");
	if (!constructor) return nullptr;
	
	// Create empty byte array (reranking doesn't return text content)
	jbyteArray byte_array = env->NewByteArray(0);
	
	// Create HashMap for probabilities (document -> score mapping)
	jclass hashmap_class = env->FindClass("java/util/HashMap");
	jmethodID hashmap_init = env->GetMethodID(hashmap_class, "<init>", "()V");
	jmethodID hashmap_put = env->GetMethodID(hashmap_class, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
	jobject probabilities = env->NewObject(hashmap_class, hashmap_init);
	
	// Process each document
	for (jsize i = 0; i < num_documents; i++) {
		jstring doc_jstr = (jstring)env->GetObjectArrayElement(documents, i);
		std::string doc_str = JniUtils::jstring_to_string(env, doc_jstr);
		
		// Tokenize document
		std::vector<llama_token> doc_tokens = tokenizeText(vocab, doc_str);
		if (doc_tokens.empty()) {
			env->DeleteLocalRef(doc_jstr);
			continue; // Skip this document
		}
		
		// Build rerank token sequence: [BOS]query[EOS][SEP]doc[EOS]
		std::vector<llama_token> rerank_tokens = buildRerankTokenSequence(vocab, query_tokens, doc_tokens);
		int total_tokens = rerank_tokens.size();
		
		// Clear previous memory for clean state
		llama_memory_clear(llama_get_memory(server->ctx), true);
		
		// Create batch for reranking computation
		llama_batch batch = llama_batch_init(total_tokens, 0, 1);
		for (int j = 0; j < total_tokens; j++) {
			batch.token[j] = rerank_tokens[j];
			batch.pos[j] = j;
			batch.n_seq_id[j] = 1;
			batch.seq_id[j][0] = 0;
			batch.logits[j] = true; // We need embeddings for reranking
		}
		batch.n_tokens = total_tokens;
		
		// Process the batch to compute reranking score
		if (llama_decode(server->ctx, batch) != 0) {
			llama_batch_free(batch);
			env->DeleteLocalRef(doc_jstr);
			continue; // Skip this document
		}
		
		// Get embeddings for reranking score
		enum llama_pooling_type pooling_type = llama_pooling_type(server->ctx);
		
		const float* embd = nullptr;
		if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
			// For reranking models, get the sequence embedding which contains the score
			embd = llama_get_embeddings_seq(server->ctx, 0);
		} else if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
			// Fallback: get embedding from last token
			embd = llama_get_embeddings_ith(server->ctx, total_tokens - 1);
		} else {
			// Other pooling types
			embd = llama_get_embeddings_seq(server->ctx, 0);
		}
		
		llama_batch_free(batch);
		
		float score = computeRerankScore(embd, pooling_type);
		
		// Add document and score to the result map
		jstring doc_key = env->NewStringUTF(doc_str.c_str());
		jclass float_class = env->FindClass("java/lang/Float");
		jmethodID float_constructor = env->GetMethodID(float_class, "<init>", "(F)V");
		jobject score_obj = env->NewObject(float_class, float_constructor, score);
		
		env->CallObjectMethod(probabilities, hashmap_put, doc_key, score_obj);
		
		env->DeleteLocalRef(doc_key);
		env->DeleteLocalRef(score_obj);
		env->DeleteLocalRef(doc_jstr);
	}
	
	return env->NewObject(output_class, constructor, byte_array, probabilities, (jboolean)true);
	
	JNI_CATCH_RET(env, nullptr)
}

// Helper function implementations
std::vector<llama_token> RerankingManager::tokenizeText(const llama_vocab* vocab, const std::string& text) {
	std::vector<llama_token> tokens;
	tokens.resize(text.length() + 1);
	
	int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(),
								  tokens.data(), tokens.size(), true, false);
	
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		n_tokens = llama_tokenize(vocab, text.c_str(), text.length(),
								  tokens.data(), tokens.size(), true, false);
	}
	
	if (n_tokens < 0) {
		return std::vector<llama_token>(); // Return empty vector on failure
	}
	
	tokens.resize(n_tokens);
	return tokens;
}

std::vector<llama_token> RerankingManager::buildRerankTokenSequence(
	const llama_vocab* vocab,
	const std::vector<llama_token>& query_tokens,
	const std::vector<llama_token>& doc_tokens) {
	
	std::vector<llama_token> rerank_tokens;
	rerank_tokens.reserve(query_tokens.size() + doc_tokens.size() + 4);
	
	// Add BOS if vocab has it
	llama_token bos_token = llama_vocab_bos(vocab);
	if (bos_token != LLAMA_TOKEN_NULL) {
		rerank_tokens.push_back(bos_token);
	}
	
	// Add query tokens
	rerank_tokens.insert(rerank_tokens.end(), query_tokens.begin(), query_tokens.end());
	
	// Add EOS token
	llama_token eos_token = llama_vocab_eos(vocab);
	if (eos_token != LLAMA_TOKEN_NULL) {
		rerank_tokens.push_back(eos_token);
	}
	
	// Add SEP token
	llama_token sep_token = llama_vocab_sep(vocab);
	if (sep_token != LLAMA_TOKEN_NULL) {
		rerank_tokens.push_back(sep_token);
	}
	
	// Add document tokens
	rerank_tokens.insert(rerank_tokens.end(), doc_tokens.begin(), doc_tokens.end());
	
	// Add final EOS token
	if (eos_token != LLAMA_TOKEN_NULL) {
		rerank_tokens.push_back(eos_token);
	}
	
	return rerank_tokens;
}

float RerankingManager::computeRerankScore(const float* embeddings, enum llama_pooling_type pooling_type) {
	if (!embeddings) {
		return 0.0f;
	}
	
	// For reranking, the score is typically the first element of the embedding
	return embeddings[0];
}