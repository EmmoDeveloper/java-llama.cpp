#pragma once

#include <jni.h>
#include <string>
#include <sstream>
#include <mutex>

/**
 * JNI-based logging system to avoid JVM channel corruption
 * Redirects native C++ logging through JVM's System.out/System.err
 */
class JNILogger {
public:
    enum Level {
        DEBUG = 0,
        INFO = 1,
        WARN = 2,
        ERROR = 3
    };

private:
    static std::mutex logger_mutex_;
    static bool initialized_;
    static Level min_level_;

    // Cached method IDs for performance
    static jclass system_class_;
    static jfieldID out_field_;
    static jfieldID err_field_;
    static jmethodID println_method_;

public:
    static JavaVM* jvm_; // Make public for JNIEnvGuard access
    // Initialize the logger with JVM reference
    static bool initialize(JNIEnv* env);
    
    // Cleanup resources
    static void shutdown(JNIEnv* env);
    
    // Set minimum log level
    static void set_level(Level level);
    
    // Log functions that route through JVM
    static void debug(const char* format, ...);
    static void info(const char* format, ...);
    static void warn(const char* format, ...);
    static void error(const char* format, ...);
    
    // Generic log function
    static void log(Level level, const char* format, ...);

private:
    static void log_internal(Level level, const std::string& message);
    static const char* level_string(Level level);
};

// Convenience macros
#ifdef DEBUG
#define JNI_LOG_DEBUG(fmt, ...) JNILogger::debug(fmt, ##__VA_ARGS__)
#else
#define JNI_LOG_DEBUG(fmt, ...)
#endif

#define JNI_LOG_INFO(fmt, ...)  JNILogger::info(fmt, ##__VA_ARGS__)
#define JNI_LOG_WARN(fmt, ...)  JNILogger::warn(fmt, ##__VA_ARGS__)
#define JNI_LOG_ERROR(fmt, ...) JNILogger::error(fmt, ##__VA_ARGS__)

// RAII class for safe JNI environment acquisition
class JNIEnvGuard {
private:
    JNIEnv* env_;
    bool needs_detach_;

public:
    JNIEnvGuard();
    ~JNIEnvGuard();
    
    JNIEnv* get() const { return env_; }
    bool is_valid() const { return env_ != nullptr; }
};