#ifndef EMBEDDING_MANAGER_H
#define EMBEDDING_MANAGER_H

#include <jni.h>
#include "llama.h"

class EmbeddingManager {
public:
	static jfloatArray createEmbedding(JNIEnv* env, jobject obj, jstring text);
	
	// Get all embeddings from context (llama_get_embeddings)
	static jfloatArray getAllEmbeddings(JNIEnv* env, jobject obj);
	
	// Set whether context outputs embeddings (llama_set_embeddings)
	static void setEmbeddingMode(JNIEnv* env, jobject obj, jboolean embeddings);

private:
	static struct llama_context* getContext(JNIEnv* env, jobject obj);
	static struct llama_model* getModel(JNIEnv* env, jobject obj);
	static bool isEmbeddingEnabled(JNIEnv* env, jobject obj);
};

#endif // EMBEDDING_MANAGER_H