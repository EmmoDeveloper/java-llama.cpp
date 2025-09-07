#ifndef MODEL_INFO_MANAGER_H
#define MODEL_INFO_MANAGER_H

#include <jni.h>
#include "llama.h"

/**
 * Model Information Manager for model introspection and vocabulary access.
 * Provides Java JNI bindings for llama.cpp model information functions.
 */
class ModelInfoManager {
public:
	// Model information functions
	static jlong getModelParameterCount(JNIEnv* env, jobject obj);
	static jlong getModelSize(JNIEnv* env, jobject obj);
	static jint getModelMetadataCount(JNIEnv* env, jobject obj);
	static jstring getModelMetadataKeyByIndex(JNIEnv* env, jobject obj, jint index);
	static jstring getModelMetadataValueByIndex(JNIEnv* env, jobject obj, jint index);
	static jstring getModelMetadataValue(JNIEnv* env, jobject obj, jstring key);
	
	// Vocabulary information functions
	static jint getVocabularyType(JNIEnv* env, jobject obj);
	static jint getVocabularySize(JNIEnv* env, jobject obj);
	static jstring getTokenText(JNIEnv* env, jobject obj, jint token);
	static jfloat getTokenScore(JNIEnv* env, jobject obj, jint token);
	static jint getTokenAttributes(JNIEnv* env, jobject obj, jint token);
	
	// Special token functions
	static jint getBosToken(JNIEnv* env, jobject obj);
	static jint getEosToken(JNIEnv* env, jobject obj);
	static jint getEotToken(JNIEnv* env, jobject obj);
	static jint getSepToken(JNIEnv* env, jobject obj);
	static jint getNlToken(JNIEnv* env, jobject obj);
	static jint getPadToken(JNIEnv* env, jobject obj);
	
	// Token checking functions
	static jboolean isEogToken(JNIEnv* env, jobject obj, jint token);
	static jboolean isControlToken(JNIEnv* env, jobject obj, jint token);

private:
	// Helper methods
	static const struct llama_model* getModel(JNIEnv* env, jobject obj);
	static const struct llama_vocab* getVocab(JNIEnv* env, jobject obj);
	static void validateModel(JNIEnv* env, const struct llama_model* model);
	static void validateVocab(JNIEnv* env, const struct llama_vocab* vocab);
	static void validateToken(JNIEnv* env, const struct llama_vocab* vocab, jint token);
};

#endif // MODEL_INFO_MANAGER_H