#include "jni_logger.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

// Static member definitions
std::mutex JNILogger::logger_mutex_;
JavaVM* JNILogger::jvm_ = nullptr;
bool JNILogger::initialized_ = false;
JNILogger::Level JNILogger::min_level_ = JNILogger::DEBUG;

jclass JNILogger::system_class_ = nullptr;
jfieldID JNILogger::out_field_ = nullptr;
jfieldID JNILogger::err_field_ = nullptr;
jmethodID JNILogger::println_method_ = nullptr;

bool JNILogger::initialize(JNIEnv* env) {
    std::lock_guard<std::mutex> lock(logger_mutex_);
    
    if (initialized_) return true;
    
    // Get JavaVM reference
    if (env->GetJavaVM(&jvm_) != JNI_OK) {
        return false;
    }
    
    // Cache System class and methods for performance
    jclass local_system_class = env->FindClass("java/lang/System");
    if (!local_system_class) return false;
    
    system_class_ = (jclass)env->NewGlobalRef(local_system_class);
    env->DeleteLocalRef(local_system_class);
    
    // Get System.out and System.err field IDs
    out_field_ = env->GetStaticFieldID(system_class_, "out", "Ljava/io/PrintStream;");
    if (!out_field_) return false;
    
    err_field_ = env->GetStaticFieldID(system_class_, "err", "Ljava/io/PrintStream;");
    if (!err_field_) return false;
    
    // Get PrintStream.println method
    jclass printstream_class = env->FindClass("java/io/PrintStream");
    if (!printstream_class) return false;
    
    println_method_ = env->GetMethodID(printstream_class, "println", "(Ljava/lang/String;)V");
    env->DeleteLocalRef(printstream_class);
    
    if (!println_method_) return false;
    
    initialized_ = true;
    return true;
}

void JNILogger::shutdown(JNIEnv* env) {
    std::lock_guard<std::mutex> lock(logger_mutex_);
    
    if (!initialized_) return;
    
    if (system_class_) {
        env->DeleteGlobalRef(system_class_);
        system_class_ = nullptr;
    }
    
    out_field_ = nullptr;
    err_field_ = nullptr;
    println_method_ = nullptr;
    jvm_ = nullptr;
    initialized_ = false;
}

void JNILogger::set_level(Level level) {
    std::lock_guard<std::mutex> lock(logger_mutex_);
    min_level_ = level;
}

void JNILogger::debug(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    log_internal(DEBUG, buffer);
}

void JNILogger::info(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    log_internal(INFO, buffer);
}

void JNILogger::warn(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    log_internal(WARN, buffer);
}

void JNILogger::error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    log_internal(ERROR, buffer);
}

void JNILogger::log(Level level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    log_internal(level, buffer);
}

void JNILogger::log_internal(Level level, const std::string& message) {
    std::lock_guard<std::mutex> lock(logger_mutex_);
    
    // Check if logging is enabled for this level
    if (level < min_level_ || !initialized_) return;
    
    JNIEnvGuard env_guard;
    JNIEnv* env = env_guard.get();
    if (!env) return;
    
    // Format message with level prefix
    std::string formatted = std::string("[") + level_string(level) + "] " + message;
    
    // Convert to Java string
    jstring java_message = env->NewStringUTF(formatted.c_str());
    if (!java_message) return;
    
    // Get appropriate output stream (stdout for INFO/DEBUG, stderr for WARN/ERROR)
    jobject output_stream;
    if (level >= WARN) {
        output_stream = env->GetStaticObjectField(system_class_, err_field_);
    } else {
        output_stream = env->GetStaticObjectField(system_class_, out_field_);
    }
    
    if (output_stream) {
        // Call println method
        env->CallVoidMethod(output_stream, println_method_, java_message);
        
        // Clear any potential exception
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }
    
    env->DeleteLocalRef(java_message);
}

const char* JNILogger::level_string(Level level) {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO:  return "INFO";
        case WARN:  return "WARN";
        case ERROR: return "ERROR";
        default:    return "UNKNOWN";
    }
}

// JNIEnvGuard implementation
JNIEnvGuard::JNIEnvGuard() : env_(nullptr), needs_detach_(false) {
    if (!JNILogger::jvm_) return;
    
    int result = JNILogger::jvm_->GetEnv((void**)&env_, JNI_VERSION_1_6);
    if (result == JNI_EDETACHED) {
        // Thread is not attached, attach it
        result = JNILogger::jvm_->AttachCurrentThread((void**)&env_, nullptr);
        if (result == JNI_OK) {
            needs_detach_ = true;
        } else {
            env_ = nullptr;
        }
    } else if (result != JNI_OK) {
        env_ = nullptr;
    }
}

JNIEnvGuard::~JNIEnvGuard() {
    if (needs_detach_ && JNILogger::jvm_) {
        JNILogger::jvm_->DetachCurrentThread();
    }
}