#ifndef BATCH_MANAGER_H
#define BATCH_MANAGER_H

#include <jni.h>
#include "llama.h"

class BatchManager {
public:
	static jlong initializeBatch(JNIEnv* env, jint tokenCount, jint embeddingSize, jint maxSequences);
	static void freeBatch(JNIEnv* env, jlong batchHandle);

	static jint encodeContext(JNIEnv* env, jobject modelObj, jlong batchHandle);
	static jint decodeTokens(JNIEnv* env, jobject modelObj, jlong batchHandle);

	static void setBatchTokens(JNIEnv* env, jlong batchHandle, jintArray tokens);
	static void setBatchEmbeddings(JNIEnv* env, jlong batchHandle, jfloatArray embeddings);
	static void setBatchPositions(JNIEnv* env, jlong batchHandle, jintArray positions);
	static void setBatchSequenceIds(JNIEnv* env, jlong batchHandle, jintArray sequenceIds);
	static void setBatchLogitFlags(JNIEnv* env, jlong batchHandle, jbyteArray logitFlags);

	static jintArray getBatchTokens(JNIEnv* env, jlong batchHandle);
	static jfloatArray getBatchEmbeddings(JNIEnv* env, jlong batchHandle);
	static jintArray getBatchPositions(JNIEnv* env, jlong batchHandle);
	static jintArray getBatchSequenceIds(JNIEnv* env, jlong batchHandle);
	static jbyteArray getBatchLogitFlags(JNIEnv* env, jlong batchHandle);

	static jint getBatchTokenCount(JNIEnv* env, jlong batchHandle);

private:
	static llama_batch* getBatch(jlong handle);
	static llama_context* getContext(JNIEnv* env, jobject modelObj);
};

#endif // BATCH_MANAGER_H