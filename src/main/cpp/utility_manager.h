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
	
	// Tier 3: Advanced system management & performance
	static jlong getContextSize(JNIEnv* env, jobject obj);
	static jlong getBatchSize(JNIEnv* env, jobject obj);
	static jlong getUbatchSize(JNIEnv* env, jobject obj);
	static jlong getMaxSequences(JNIEnv* env, jobject obj);
	static jlong getCurrentThreads(JNIEnv* env, jobject obj);
	static jlong getCurrentThreadsBatch(JNIEnv* env, jobject obj);
	static void attachThreadPool(JNIEnv* env, jobject obj, jlong threadpool, jlong threadpool_batch);
	static void detachThreadPool(JNIEnv* env, jobject obj);
	
	// Tier 4: Performance monitoring & model architecture
	static jstring getPerformanceData(JNIEnv* env, jobject obj);
	static void printPerformanceData(JNIEnv* env, jobject obj);
	static void resetPerformanceData(JNIEnv* env, jobject obj);
	static jlong getModelLayerCount(JNIEnv* env, jobject obj);
	static jlong getModelTrainingContextSize(JNIEnv* env, jobject obj);
	static jboolean hasEncoder(JNIEnv* env, jobject obj);
	static jboolean hasDecoder(JNIEnv* env, jobject obj);
	static jint getRopeType(JNIEnv* env, jobject obj);
	static jfloat getRopeFrequencyScale(JNIEnv* env, jobject obj);
	
	// Tier 5: Advanced model introspection & resource control
	static jlong getModelEmbeddingDimension(JNIEnv* env, jobject obj);
	static jlong getModelAttentionHeads(JNIEnv* env, jobject obj);
	static jlong getModelKeyValueHeads(JNIEnv* env, jobject obj);
	static jboolean isRecurrentModel(JNIEnv* env, jobject obj);
	static jboolean isDiffusionModel(JNIEnv* env, jobject obj);
	static void setWarmupMode(JNIEnv* env, jobject obj, jboolean warmup);
	static jstring getFlashAttentionType(JNIEnv* env, jobject obj);
	static void initializeBackend(JNIEnv* env, jclass cls);
	static void freeBackend(JNIEnv* env, jclass cls);
	static void initializeNuma(JNIEnv* env, jclass cls, jint strategy);
};

#endif // UTILITY_MANAGER_H