#ifndef KV_CACHE_MANAGER_H
#define KV_CACHE_MANAGER_H

#include <jni.h>
#include "llama.h"

/**
 * KV Cache Manager for advanced memory operations.
 * Provides Java JNI bindings for llama.cpp KV cache memory management functions.
 */
class KVCacheManager {
public:
	// Sequence copying and manipulation
	static void copySequence(JNIEnv* env, jobject obj, jint srcSeqId, jint dstSeqId, jint p0, jint p1);
	static void keepSequence(JNIEnv* env, jobject obj, jint seqId);
	static void addPositionDelta(JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1, jint delta);
	static void dividePositions(JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1, jint divisor);
	
	// Sequence position queries
	static jint getSequenceMinPosition(JNIEnv* env, jobject obj, jint seqId);
	static jint getSequenceMaxPosition(JNIEnv* env, jobject obj, jint seqId);
	
	// Memory capabilities
	static jboolean canShiftContext(JNIEnv* env, jobject obj);
	
	// Memory clearing (expose existing function as public API)
	static void clearMemory(JNIEnv* env, jobject obj, jboolean clearData);
	
	// Sequence removal (expose existing function as public API)
	static jboolean removeSequenceTokens(JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1);

private:
	// Helper methods
	static struct llama_context* getContext(JNIEnv* env, jobject obj);
	static llama_memory_t getMemory(JNIEnv* env, jobject obj);
	static void validateSequenceId(JNIEnv* env, jint seqId);
	static void validatePosition(JNIEnv* env, jint position);
};

#endif // KV_CACHE_MANAGER_H