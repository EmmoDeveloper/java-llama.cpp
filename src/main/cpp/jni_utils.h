#pragma once

#include <jni.h>
#include <string>

// JNI utility functions for string conversion and common operations
class JniUtils {
public:
    static std::string jstring_to_string(JNIEnv* env, jstring jstr);
    static jstring string_to_jstring(JNIEnv* env, const std::string& str);
};