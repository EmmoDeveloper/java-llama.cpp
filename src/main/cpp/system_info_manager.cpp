#include "system_info_manager.h"
#include "jni_utils.h"

jstring SystemInfoManager::getSystemInfo(JNIEnv* env) {
	const char* info = llama_print_system_info();
	return env->NewStringUTF(info);
}

jlong SystemInfoManager::getTimeUs(JNIEnv* env) {
	return static_cast<jlong>(llama_time_us());
}

jboolean SystemInfoManager::supportsMmap(JNIEnv* env) {
	return llama_supports_mmap() ? JNI_TRUE : JNI_FALSE;
}

jboolean SystemInfoManager::supportsMlock(JNIEnv* env) {
	return llama_supports_mlock() ? JNI_TRUE : JNI_FALSE;
}

jboolean SystemInfoManager::supportsGpuOffload(JNIEnv* env) {
	return llama_supports_gpu_offload() ? JNI_TRUE : JNI_FALSE;
}

jboolean SystemInfoManager::supportsRpc(JNIEnv* env) {
	return llama_supports_rpc() ? JNI_TRUE : JNI_FALSE;
}