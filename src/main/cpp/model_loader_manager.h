#ifndef MODEL_LOADER_MANAGER_H
#define MODEL_LOADER_MANAGER_H

#include <jni.h>

class ModelLoaderManager {
public:
	// Load model from multiple split files (llama_model_load_from_splits)
	static jlong loadModelFromSplits(JNIEnv* env, jclass cls, jobjectArray paths, jobject params);
	
	// Save model to file (llama_model_save_to_file) 
	static void saveModelToFile(JNIEnv* env, jobject obj, jstring path);
};

#endif // MODEL_LOADER_MANAGER_H