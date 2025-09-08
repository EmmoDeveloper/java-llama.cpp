#ifndef RERANKING_MANAGER_H
#define RERANKING_MANAGER_H

#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"

class RerankingManager {
public:
	// Rerank documents against a query
	static jobject rerank(JNIEnv* env, jobject obj, jstring query, jobjectArray documents);

private:
	// Helper functions for reranking
	static std::vector<llama_token> tokenizeText(const llama_vocab* vocab, const std::string& text);
	static std::vector<llama_token> buildRerankTokenSequence(
		const llama_vocab* vocab,
		const std::vector<llama_token>& query_tokens,
		const std::vector<llama_token>& doc_tokens
	);
	static float computeRerankScore(const float* embeddings, enum llama_pooling_type pooling_type);
};

#endif // RERANKING_MANAGER_H