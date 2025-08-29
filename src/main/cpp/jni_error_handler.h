#pragma once

#include <jni.h>
#include <string>
#include <exception>
#include <stdexcept>
#include <functional>
#include "jni_logger.h"

/**
 * Comprehensive JNI error handling system
 * Provides unified error management for native code
 */

// Custom exception types for JNI operations
class JNIException : public std::runtime_error {
public:
    explicit JNIException(const std::string& msg) : std::runtime_error(msg) {}
};

class ModelLoadException : public JNIException {
public:
    explicit ModelLoadException(const std::string& msg) 
        : JNIException("Model loading failed: " + msg) {}
};

class ContextCreationException : public JNIException {
public:
    explicit ContextCreationException(const std::string& msg) 
        : JNIException("Context creation failed: " + msg) {}
};

class InferenceException : public JNIException {
public:
    explicit InferenceException(const std::string& msg) 
        : JNIException("Inference failed: " + msg) {}
};

// JNI Error Handler class
class JNIErrorHandler {
private:
    static thread_local std::string last_error_;
    static thread_local bool has_pending_exception_;

public:
    // Check if JNI environment has a pending exception
    static bool check_exception(JNIEnv* env);
    
    // Clear any pending exception and log it
    static void clear_exception(JNIEnv* env);
    
    // Throw a Java exception with the given message
    static void throw_java_exception(JNIEnv* env, const char* class_name, const std::string& message);
    
    // Common exception throwers
    static void throw_runtime_exception(JNIEnv* env, const std::string& message);
    static void throw_illegal_argument(JNIEnv* env, const std::string& message);
    static void throw_illegal_state(JNIEnv* env, const std::string& message);
    static void throw_out_of_memory(JNIEnv* env, const std::string& message);
    static void throw_null_pointer(JNIEnv* env, const std::string& message);
    
    // Handle native exceptions and convert to Java exceptions
    static void handle_native_exception(JNIEnv* env, const std::exception& e);
    static void handle_unknown_exception(JNIEnv* env);
    
    // Get and clear last error message
    static std::string get_last_error();
    static void set_last_error(const std::string& error);
    
    // Check for null pointers and throw appropriate exception
    template<typename T>
    static bool check_null(JNIEnv* env, T* ptr, const std::string& name) {
        if (!ptr) {
            throw_null_pointer(env, name + " is null");
            return true;
        }
        return false;
    }
    
    // Validate string parameters
    static bool validate_string(JNIEnv* env, jstring str, const std::string& param_name);
    
    // Validate array parameters
    static bool validate_array(JNIEnv* env, jarray arr, const std::string& param_name, jsize min_length = 0);
};

// RAII wrapper for exception-safe JNI operations
class JNIExceptionGuard {
private:
    JNIEnv* env_;
    bool had_exception_;
    
public:
    explicit JNIExceptionGuard(JNIEnv* env) : env_(env), had_exception_(false) {
        had_exception_ = JNIErrorHandler::check_exception(env);
        if (had_exception_) {
            JNIErrorHandler::clear_exception(env);
        }
    }
    
    ~JNIExceptionGuard() {
        if (!had_exception_ && JNIErrorHandler::check_exception(env_)) {
            // New exception occurred during our operation
            JNI_LOG_ERROR("Unhandled exception detected in JNI operation");
        }
    }
    
    bool had_prior_exception() const { return had_exception_; }
};

// Macros for common error checking patterns
#define JNI_CHECK_NULL(env, ptr, name) \
    if (JNIErrorHandler::check_null(env, ptr, name)) return nullptr

#define JNI_CHECK_NULL_VOID(env, ptr, name) \
    if (JNIErrorHandler::check_null(env, ptr, name)) return

#define JNI_CHECK_NULL_RET(env, ptr, name, ret) \
    if (JNIErrorHandler::check_null(env, ptr, name)) return ret

#define JNI_TRY(env) \
    try {

#define JNI_CATCH(env) \
    } catch (const std::exception& e) { \
        JNIErrorHandler::handle_native_exception(env, e); \
    } catch (...) { \
        JNIErrorHandler::handle_unknown_exception(env); \
    }

#define JNI_CATCH_RET(env, ret) \
    } catch (const std::exception& e) { \
        JNIErrorHandler::handle_native_exception(env, e); \
        return ret; \
    } catch (...) { \
        JNIErrorHandler::handle_unknown_exception(env); \
        return ret; \
    }

// Safe resource management helpers
template<typename T>
class JNIScopedResource {
private:
    T* resource_;
    std::function<void(T*)> deleter_;
    
public:
    JNIScopedResource(T* resource, std::function<void(T*)> deleter)
        : resource_(resource), deleter_(deleter) {}
    
    ~JNIScopedResource() {
        if (resource_ && deleter_) {
            deleter_(resource_);
        }
    }
    
    T* get() const { return resource_; }
    T* operator->() const { return resource_; }
    T& operator*() const { return *resource_; }
    
    T* release() {
        T* temp = resource_;
        resource_ = nullptr;
        return temp;
    }
    
    // Delete copy constructor and assignment
    JNIScopedResource(const JNIScopedResource&) = delete;
    JNIScopedResource& operator=(const JNIScopedResource&) = delete;
};

// Error context for better debugging
class ErrorContext {
private:
    std::string operation_;
    std::string details_;
    
public:
    ErrorContext(const std::string& op) : operation_(op) {}
    
    ErrorContext& with_detail(const std::string& key, const std::string& value) {
        if (!details_.empty()) details_ += ", ";
        details_ += key + "=" + value;
        return *this;
    }
    
    ErrorContext& with_detail(const std::string& key, int value) {
        return with_detail(key, std::to_string(value));
    }
    
    std::string build_message(const std::string& error) const {
        std::string msg = "Operation '" + operation_ + "' failed: " + error;
        if (!details_.empty()) {
            msg += " (" + details_ + ")";
        }
        return msg;
    }
};