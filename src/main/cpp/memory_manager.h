#pragma once

#include <jni.h>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <string>
#include "llama.h"

// RAII wrapper for llama_batch to ensure proper cleanup
class BatchRAII {
private:
    llama_batch batch_;
    bool initialized_;

public:
    BatchRAII(int n_tokens, int embd, int n_seq_max) 
        : batch_(llama_batch_init(n_tokens, embd, n_seq_max)), initialized_(true) {}
    
    ~BatchRAII() {
        if (initialized_) {
            llama_batch_free(batch_);
        }
    }
    
    // Move constructor
    BatchRAII(BatchRAII&& other) noexcept 
        : batch_(other.batch_), initialized_(other.initialized_) {
        other.initialized_ = false;
    }
    
    // Delete copy constructor and assignment
    BatchRAII(const BatchRAII&) = delete;
    BatchRAII& operator=(const BatchRAII&) = delete;
    BatchRAII& operator=(BatchRAII&&) = delete;
    
    llama_batch* get() { return &batch_; }
    llama_batch& operator*() { return batch_; }
    llama_batch* operator->() { return &batch_; }
};

// RAII wrapper for JNI local reference management
class JNILocalFrameRAII {
private:
    JNIEnv* env_;
    bool pushed_;

public:
    JNILocalFrameRAII(JNIEnv* env, jint capacity = 16) : env_(env), pushed_(false) {
        if (env->PushLocalFrame(capacity) == JNI_OK) {
            pushed_ = true;
        }
    }
    
    ~JNILocalFrameRAII() {
        if (pushed_) {
            env_->PopLocalFrame(nullptr);
        }
    }
    
    // Delete copy/move constructors and assignment operators
    JNILocalFrameRAII(const JNILocalFrameRAII&) = delete;
    JNILocalFrameRAII(JNILocalFrameRAII&&) = delete;
    JNILocalFrameRAII& operator=(const JNILocalFrameRAII&) = delete;
    JNILocalFrameRAII& operator=(JNILocalFrameRAII&&) = delete;
    
    bool isValid() const { return pushed_; }
};

// Memory usage tracking for leak detection
class MemoryTracker {
private:
    struct AllocationInfo {
        size_t size;
        std::chrono::steady_clock::time_point timestamp;
        std::string file;
        int line;
    };
    
    static std::mutex tracker_mutex_;
    static std::unordered_map<void*, AllocationInfo> allocations_;
    static size_t total_allocated_;
    static size_t peak_usage_;
    static bool tracking_enabled_;

public:
    static void enable_tracking(bool enable = true);
    static void track_allocation(void* ptr, size_t size, const char* file, int line);
    static void track_deallocation(void* ptr);
    static size_t get_current_usage();
    static size_t get_peak_usage();
    static size_t get_allocation_count();
    static void print_leak_report();
    static void reset_stats();
};

// Memory pool for frequent small allocations
class MemoryPool {
private:
    struct Block {
        std::unique_ptr<char[]> memory;
        size_t size;
        size_t offset;
        
        Block(size_t sz) : memory(std::make_unique<char[]>(sz)), size(sz), offset(0) {}
    };
    
    std::vector<std::unique_ptr<Block>> blocks_;
    std::mutex pool_mutex_;
    static const size_t DEFAULT_BLOCK_SIZE = 64 * 1024; // 64KB blocks
    
public:
    void* allocate(size_t size);
    void reset(); // Reset all blocks but keep memory allocated
    ~MemoryPool();
};

// Global memory pool instance
extern thread_local MemoryPool g_memory_pool;

// Macros for tracked allocation
#ifdef DEBUG_MEMORY
#define TRACKED_MALLOC(size) tracked_malloc(size, __FILE__, __LINE__)
#define TRACKED_FREE(ptr) tracked_free(ptr, __FILE__, __LINE__)

void* tracked_malloc(size_t size, const char* file, int line);
void tracked_free(void* ptr, const char* file, int line);
#else
#define TRACKED_MALLOC(size) malloc(size)
#define TRACKED_FREE(ptr) free(ptr)
#endif

// Utility functions for safe JNI memory management
namespace JNIMemUtils {
    // Safe array element access with cleanup
    template<typename T>
    class SafeArrayElements {
    private:
        JNIEnv* env_;
        jarray array_;
        T* elements_;
        jint mode_;
        
    public:
        SafeArrayElements(JNIEnv* env, jarray array, jint mode = 0);
        ~SafeArrayElements();
        
        T* get() const { return elements_; }
        T* operator->() const { return elements_; }
        T& operator[](size_t index) const { return elements_[index]; }
        
        // Delete copy/move constructors and assignment operators
        SafeArrayElements(const SafeArrayElements&) = delete;
        SafeArrayElements(SafeArrayElements&&) = delete;
        SafeArrayElements& operator=(const SafeArrayElements&) = delete;
        SafeArrayElements& operator=(SafeArrayElements&&) = delete;
    };
    
    // Safe string conversion with cleanup
    class SafeString {
    private:
        JNIEnv* env_;
        jstring jstr_;
        const char* str_;
        
    public:
        SafeString(JNIEnv* env, jstring jstr);
        ~SafeString();
        
        const char* c_str() const { return str_; }
        std::string str() const { return str_ ? std::string(str_) : std::string(); }
        
        // Delete copy/move constructors and assignment operators
        SafeString(const SafeString&) = delete;
        SafeString(SafeString&&) = delete;
        SafeString& operator=(const SafeString&) = delete;
        SafeString& operator=(SafeString&&) = delete;
    };
}