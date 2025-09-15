#ifndef SYSTEM_INFO_MANAGER_H
#define SYSTEM_INFO_MANAGER_H

#include <jni.h>
#include "llama.h"
#include <string>

class SystemInfoManager {
public:
	static jstring getSystemInfo(JNIEnv* env);
	static jlong getTimeUs(JNIEnv* env);
	static jboolean supportsMmap(JNIEnv* env);
	static jboolean supportsMlock(JNIEnv* env);
	static jboolean supportsGpuOffload(JNIEnv* env);
	static jboolean supportsRpc(JNIEnv* env);
};

#endif // SYSTEM_INFO_MANAGER_H