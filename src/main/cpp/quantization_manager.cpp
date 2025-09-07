#include "quantization_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include <string>

jobject QuantizationManager::getDefaultQuantizationParams(JNIEnv* env) {
	JNI_TRY(env)

	llama_model_quantize_params defaultParams = llama_model_quantize_default_params();
	return createJavaParams(env, defaultParams);

	JNI_CATCH_RET(env, nullptr)
}

jint QuantizationManager::quantizeModel(JNIEnv* env, jstring inputPath, jstring outputPath, jobject params) {
	JNI_TRY(env)

	if (!inputPath || !outputPath) {
		JNIErrorHandler::throw_illegal_argument(env, "Input and output paths cannot be null");
		return -1;
	}

	const char* inputStr = env->GetStringUTFChars(inputPath, nullptr);
	if (!inputStr) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get input path string");
		return -1;
	}

	const char* outputStr = env->GetStringUTFChars(outputPath, nullptr);
	if (!outputStr) {
		env->ReleaseStringUTFChars(inputPath, inputStr);
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get output path string");
		return -1;
	}

	llama_model_quantize_params quantizeParams;
	if (params) {
		quantizeParams = convertJavaParams(env, params);
		if (env->ExceptionCheck()) {
			env->ReleaseStringUTFChars(inputPath, inputStr);
			env->ReleaseStringUTFChars(outputPath, outputStr);
			return -1;
		}
	} else {
		quantizeParams = llama_model_quantize_default_params();
	}

	uint32_t result = llama_model_quantize(inputStr, outputStr, &quantizeParams);

	env->ReleaseStringUTFChars(inputPath, inputStr);
	env->ReleaseStringUTFChars(outputPath, outputStr);

	return static_cast<jint>(result);

	JNI_CATCH_RET(env, -1)
}

llama_model_quantize_params QuantizationManager::convertJavaParams(JNIEnv* env, jobject javaParams) {
	llama_model_quantize_params params = llama_model_quantize_default_params();

	if (!javaParams) {
		return params;
	}

	jclass paramsClass = env->GetObjectClass(javaParams);

	// Get nthread field
	jfieldID nthreadField = env->GetFieldID(paramsClass, "nthread", "I");
	if (nthreadField) {
		params.nthread = env->GetIntField(javaParams, nthreadField);
	}

	// Get ftype field (as int)
	jfieldID ftypeField = env->GetFieldID(paramsClass, "ftype", "I");
	if (ftypeField) {
		int ftype = env->GetIntField(javaParams, ftypeField);
		params.ftype = static_cast<llama_ftype>(ftype);
	}

	// Get boolean fields
	jfieldID allowRequantizeField = env->GetFieldID(paramsClass, "allowRequantize", "Z");
	if (allowRequantizeField) {
		params.allow_requantize = env->GetBooleanField(javaParams, allowRequantizeField);
	}

	jfieldID quantizeOutputField = env->GetFieldID(paramsClass, "quantizeOutputTensor", "Z");
	if (quantizeOutputField) {
		params.quantize_output_tensor = env->GetBooleanField(javaParams, quantizeOutputField);
	}

	jfieldID onlyCopyField = env->GetFieldID(paramsClass, "onlyCopy", "Z");
	if (onlyCopyField) {
		params.only_copy = env->GetBooleanField(javaParams, onlyCopyField);
	}

	jfieldID pureField = env->GetFieldID(paramsClass, "pure", "Z");
	if (pureField) {
		params.pure = env->GetBooleanField(javaParams, pureField);
	}

	jfieldID keepSplitField = env->GetFieldID(paramsClass, "keepSplit", "Z");
	if (keepSplitField) {
		params.keep_split = env->GetBooleanField(javaParams, keepSplitField);
	}

	return params;
}

jobject QuantizationManager::createJavaParams(JNIEnv* env, const llama_model_quantize_params& params) {
	jclass paramsClass = env->FindClass("de/kherud/llama/LlamaQuantizer$QuantizationParams");
	if (!paramsClass) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to find QuantizationParams class");
		return nullptr;
	}

	jmethodID constructor = env->GetMethodID(paramsClass, "<init>", "()V");
	if (!constructor) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to find QuantizationParams constructor");
		return nullptr;
	}

	jobject javaParams = env->NewObject(paramsClass, constructor);
	if (!javaParams) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create QuantizationParams object");
		return nullptr;
	}

	// Set nthread field
	jfieldID nthreadField = env->GetFieldID(paramsClass, "nthread", "I");
	if (nthreadField) {
		env->SetIntField(javaParams, nthreadField, params.nthread);
	}

	// Set ftype field
	jfieldID ftypeField = env->GetFieldID(paramsClass, "ftype", "I");
	if (ftypeField) {
		env->SetIntField(javaParams, ftypeField, static_cast<int>(params.ftype));
	}

	// Set boolean fields
	jfieldID allowRequantizeField = env->GetFieldID(paramsClass, "allowRequantize", "Z");
	if (allowRequantizeField) {
		env->SetBooleanField(javaParams, allowRequantizeField, params.allow_requantize);
	}

	jfieldID quantizeOutputField = env->GetFieldID(paramsClass, "quantizeOutputTensor", "Z");
	if (quantizeOutputField) {
		env->SetBooleanField(javaParams, quantizeOutputField, params.quantize_output_tensor);
	}

	jfieldID onlyCopyField = env->GetFieldID(paramsClass, "onlyCopy", "Z");
	if (onlyCopyField) {
		env->SetBooleanField(javaParams, onlyCopyField, params.only_copy);
	}

	jfieldID pureField = env->GetFieldID(paramsClass, "pure", "Z");
	if (pureField) {
		env->SetBooleanField(javaParams, pureField, params.pure);
	}

	jfieldID keepSplitField = env->GetFieldID(paramsClass, "keepSplit", "Z");
	if (keepSplitField) {
		env->SetBooleanField(javaParams, keepSplitField, params.keep_split);
	}

	return javaParams;
}