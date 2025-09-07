#ifndef QUANTIZATION_MANAGER_H
#define QUANTIZATION_MANAGER_H

#include <jni.h>
#include "llama.h"

class QuantizationManager {
public:
	static jobject getDefaultQuantizationParams(JNIEnv* env);
	static jint quantizeModel(JNIEnv* env, jstring inputPath, jstring outputPath, jobject params);

private:
	static llama_model_quantize_params convertJavaParams(JNIEnv* env, jobject javaParams);
	static jobject createJavaParams(JNIEnv* env, const llama_model_quantize_params& params);
};

#endif // QUANTIZATION_MANAGER_H