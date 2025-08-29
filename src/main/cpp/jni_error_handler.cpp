#include "jni_error_handler.h"
#include <sstream>
#include <iomanip>

// Static member definitions
thread_local std::string JNIErrorHandler::last_error_;
thread_local bool JNIErrorHandler::has_pending_exception_ = false;

bool JNIErrorHandler::check_exception(JNIEnv* env) {
    if (!env) return false;
    return env->ExceptionCheck() == JNI_TRUE;
}

void JNIErrorHandler::clear_exception(JNIEnv* env) {
    if (!env) return;
    
    if (check_exception(env)) {
        // Get exception details before clearing
        jthrowable exception = env->ExceptionOccurred();
        env->ExceptionClear();
        
        if (exception) {
            // Try to get exception message
            jclass throwable_class = env->FindClass("java/lang/Throwable");
            if (throwable_class) {
                jmethodID get_message = env->GetMethodID(throwable_class, "getMessage", "()Ljava/lang/String;");
                jmethodID get_class = env->GetMethodID(throwable_class, "getClass", "()Ljava/lang/Class;");
                
                if (get_message && get_class) {
                    jstring message = (jstring)env->CallObjectMethod(exception, get_message);
                    jobject exc_class = env->CallObjectMethod(exception, get_class);
                    
                    if (exc_class) {
                        jclass class_class = env->FindClass("java/lang/Class");
                        jmethodID get_name = env->GetMethodID(class_class, "getName", "()Ljava/lang/String;");
                        jstring class_name = (jstring)env->CallObjectMethod(exc_class, get_name);
                        
                        if (class_name) {
                            const char* class_str = env->GetStringUTFChars(class_name, nullptr);
                            const char* msg_str = message ? env->GetStringUTFChars(message, nullptr) : "No message";
                            
                            JNI_LOG_ERROR("Cleared Java exception: %s: %s", class_str, msg_str);
                            
                            env->ReleaseStringUTFChars(class_name, class_str);
                            if (message) env->ReleaseStringUTFChars(message, msg_str);
                        }
                    }
                    
                    if (message) env->DeleteLocalRef(message);
                    if (exc_class) env->DeleteLocalRef(exc_class);
                }
                env->DeleteLocalRef(throwable_class);
            }
            env->DeleteLocalRef(exception);
        }
    }
}

void JNIErrorHandler::throw_java_exception(JNIEnv* env, const char* class_name, const std::string& message) {
    if (!env) return;
    
    // Clear any pending exception first
    clear_exception(env);
    
    jclass exception_class = env->FindClass(class_name);
    if (exception_class) {
        env->ThrowNew(exception_class, message.c_str());
        env->DeleteLocalRef(exception_class);
        has_pending_exception_ = true;
        last_error_ = message;
        JNI_LOG_ERROR("Throwing %s: %s", class_name, message.c_str());
    } else {
        // Fallback to RuntimeException if we can't find the requested class
        jclass runtime_class = env->FindClass("java/lang/RuntimeException");
        if (runtime_class) {
            std::string fallback_msg = "Failed to find exception class " + std::string(class_name) + ": " + message;
            env->ThrowNew(runtime_class, fallback_msg.c_str());
            env->DeleteLocalRef(runtime_class);
            has_pending_exception_ = true;
            last_error_ = fallback_msg;
            JNI_LOG_ERROR("Throwing RuntimeException (fallback): %s", fallback_msg.c_str());
        }
    }
}

void JNIErrorHandler::throw_runtime_exception(JNIEnv* env, const std::string& message) {
    throw_java_exception(env, "java/lang/RuntimeException", message);
}

void JNIErrorHandler::throw_illegal_argument(JNIEnv* env, const std::string& message) {
    throw_java_exception(env, "java/lang/IllegalArgumentException", message);
}

void JNIErrorHandler::throw_illegal_state(JNIEnv* env, const std::string& message) {
    throw_java_exception(env, "java/lang/IllegalStateException", message);
}

void JNIErrorHandler::throw_out_of_memory(JNIEnv* env, const std::string& message) {
    throw_java_exception(env, "java/lang/OutOfMemoryError", message);
}

void JNIErrorHandler::throw_null_pointer(JNIEnv* env, const std::string& message) {
    throw_java_exception(env, "java/lang/NullPointerException", message);
}

void JNIErrorHandler::handle_native_exception(JNIEnv* env, const std::exception& e) {
    std::string error_msg = "Native exception: ";
    error_msg += e.what();
    
    // Check for specific exception types
    if (dynamic_cast<const ModelLoadException*>(&e)) {
        throw_runtime_exception(env, error_msg);
    } else if (dynamic_cast<const ContextCreationException*>(&e)) {
        throw_runtime_exception(env, error_msg);
    } else if (dynamic_cast<const InferenceException*>(&e)) {
        throw_runtime_exception(env, error_msg);
    } else if (dynamic_cast<const std::bad_alloc*>(&e)) {
        throw_out_of_memory(env, "Native memory allocation failed");
    } else if (dynamic_cast<const std::invalid_argument*>(&e)) {
        throw_illegal_argument(env, error_msg);
    } else {
        throw_runtime_exception(env, error_msg);
    }
}

void JNIErrorHandler::handle_unknown_exception(JNIEnv* env) {
    throw_runtime_exception(env, "Unknown native exception occurred");
}

std::string JNIErrorHandler::get_last_error() {
    std::string error = last_error_;
    last_error_.clear();
    return error;
}

void JNIErrorHandler::set_last_error(const std::string& error) {
    last_error_ = error;
    JNI_LOG_ERROR("Error set: %s", error.c_str());
}

bool JNIErrorHandler::validate_string(JNIEnv* env, jstring str, const std::string& param_name) {
    if (!str) {
        throw_null_pointer(env, param_name + " string parameter is null");
        return false;
    }
    
    // Check if we can get the string chars (this can fail if string is corrupted)
    const char* chars = env->GetStringUTFChars(str, nullptr);
    if (!chars) {
        throw_runtime_exception(env, "Failed to get UTF chars from " + param_name);
        return false;
    }
    
    env->ReleaseStringUTFChars(str, chars);
    return true;
}

bool JNIErrorHandler::validate_array(JNIEnv* env, jarray arr, const std::string& param_name, jsize min_length) {
    if (!arr) {
        throw_null_pointer(env, param_name + " array parameter is null");
        return false;
    }
    
    jsize length = env->GetArrayLength(arr);
    if (check_exception(env)) {
        throw_runtime_exception(env, "Failed to get array length for " + param_name);
        return false;
    }
    
    if (length < min_length) {
        std::stringstream ss;
        ss << param_name << " array too small: " << length << " < " << min_length;
        throw_illegal_argument(env, ss.str());
        return false;
    }
    
    return true;
}