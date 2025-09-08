#ifndef UTILITY_MANAGER_H
#define UTILITY_MANAGER_H

#include <jni.h>

class UtilityManager {
public:
	// System capability functions
	static jboolean supportsGpuOffload(JNIEnv* env, jclass cls);
	static jboolean supportsMmap(JNIEnv* env, jclass cls);
	static jboolean supportsMlock(JNIEnv* env, jclass cls);
	static jboolean supportsRpc(JNIEnv* env, jclass cls);
	static jlong maxDevices(JNIEnv* env, jclass cls);
	static jlong maxParallelSequences(JNIEnv* env, jclass cls);
	
	// System information
	static jstring printSystemInfo(JNIEnv* env, jclass cls);
	
	// Performance timing
	static jlong timeUs(JNIEnv* env, jclass cls);
	
	// Logging control
	static void setLogCallback(JNIEnv* env, jclass cls, jobject callback);
	
	// Abort callback for long operations
	static void setAbortCallback(JNIEnv* env, jobject obj, jobject callback);
	
	// Tier 2: Operational improvements
	static void setThreadCount(JNIEnv* env, jobject obj, jint threads);
	static void synchronizeOperations(JNIEnv* env, jobject obj);
	static void setEmbeddingMode(JNIEnv* env, jobject obj, jboolean embeddings);
	static void setCausalAttention(JNIEnv* env, jobject obj, jboolean causal);
	static jstring splitPath(JNIEnv* env, jclass cls, jstring path, jint split);
};

#endif // UTILITY_MANAGER_H