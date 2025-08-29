#include "memory_manager.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

// Static member definitions
std::mutex MemoryTracker::tracker_mutex_;
std::unordered_map<void*, MemoryTracker::AllocationInfo> MemoryTracker::allocations_;
size_t MemoryTracker::total_allocated_ = 0;
size_t MemoryTracker::peak_usage_ = 0;
bool MemoryTracker::tracking_enabled_ = false;

// Thread-local memory pool
thread_local MemoryPool g_memory_pool;

// MemoryTracker implementation
void MemoryTracker::enable_tracking(bool enable) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    tracking_enabled_ = enable;
    if (enable) {
        allocations_.clear();
        total_allocated_ = 0;
        peak_usage_ = 0;
    }
}

void MemoryTracker::track_allocation(void* ptr, size_t size, const char* file, int line) {
    if (!tracking_enabled_ || !ptr) return;
    
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    AllocationInfo info;
    info.size = size;
    info.timestamp = std::chrono::steady_clock::now();
    info.file = file ? file : "unknown";
    info.line = line;
    allocations_[ptr] = info;
    
    total_allocated_ += size;
    if (total_allocated_ > peak_usage_) {
        peak_usage_ = total_allocated_;
    }
}

void MemoryTracker::track_deallocation(void* ptr) {
    if (!tracking_enabled_ || !ptr) return;
    
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        total_allocated_ -= it->second.size;
        allocations_.erase(it);
    }
}

size_t MemoryTracker::get_current_usage() {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    return total_allocated_;
}

size_t MemoryTracker::get_peak_usage() {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    return peak_usage_;
}

size_t MemoryTracker::get_allocation_count() {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    return allocations_.size();
}

void MemoryTracker::print_leak_report() {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    if (allocations_.empty()) {
        std::cout << "✅ No memory leaks detected!" << std::endl;
        std::cout << "Peak memory usage: " << peak_usage_ << " bytes" << std::endl;
        return;
    }
    
    std::cout << "🚨 MEMORY LEAKS DETECTED!" << std::endl;
    std::cout << "Total leaked: " << total_allocated_ << " bytes in " << allocations_.size() << " allocations" << std::endl;
    std::cout << "Peak usage: " << peak_usage_ << " bytes" << std::endl;
    std::cout << "\nLeak details:" << std::endl;
    
    auto now = std::chrono::steady_clock::now();
    for (const auto& [ptr, info] : allocations_) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - info.timestamp);
        std::cout << "  - " << info.size << " bytes at " << ptr 
                  << " (allocated " << duration.count() << "ms ago)"
                  << " [" << info.file << ":" << info.line << "]" << std::endl;
    }
}

void MemoryTracker::reset_stats() {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    allocations_.clear();
    total_allocated_ = 0;
    peak_usage_ = 0;
}

// MemoryPool implementation
void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Find a block with enough space
    for (auto& block : blocks_) {
        if (block->offset + size <= block->size) {
            void* ptr = block->memory.get() + block->offset;
            block->offset += size;
            return ptr;
        }
    }
    
    // Need a new block
    size_t block_size = std::max(size, DEFAULT_BLOCK_SIZE);
    auto new_block = std::make_unique<Block>(block_size);
    void* ptr = new_block->memory.get();
    new_block->offset = size;
    blocks_.push_back(std::move(new_block));
    
    return ptr;
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto& block : blocks_) {
        block->offset = 0;
    }
}

MemoryPool::~MemoryPool() {
    // Blocks will be automatically cleaned up by unique_ptr
}

// Tracked allocation functions
#ifdef DEBUG_MEMORY
void* tracked_malloc(size_t size, const char* file, int line) {
    void* ptr = malloc(size);
    MemoryTracker::track_allocation(ptr, size, file, line);
    return ptr;
}

void tracked_free(void* ptr, const char* file, int line) {
    MemoryTracker::track_deallocation(ptr);
    free(ptr);
}
#endif

// JNIMemUtils implementation
namespace JNIMemUtils {
    
    // SafeArrayElements specializations
    template<>
    SafeArrayElements<jint>::SafeArrayElements(JNIEnv* env, jarray array, jint mode) 
        : env_(env), array_(array), mode_(mode) {
        elements_ = env_->GetIntArrayElements(static_cast<jintArray>(array_), nullptr);
    }
    
    template<>
    SafeArrayElements<jint>::~SafeArrayElements() {
        if (elements_) {
            env_->ReleaseIntArrayElements(static_cast<jintArray>(array_), elements_, mode_);
        }
    }
    
    template<>
    SafeArrayElements<jbyte>::SafeArrayElements(JNIEnv* env, jarray array, jint mode) 
        : env_(env), array_(array), mode_(mode) {
        elements_ = env_->GetByteArrayElements(static_cast<jbyteArray>(array_), nullptr);
    }
    
    template<>
    SafeArrayElements<jbyte>::~SafeArrayElements() {
        if (elements_) {
            env_->ReleaseByteArrayElements(static_cast<jbyteArray>(array_), elements_, mode_);
        }
    }
    
    template<>
    SafeArrayElements<jfloat>::SafeArrayElements(JNIEnv* env, jarray array, jint mode) 
        : env_(env), array_(array), mode_(mode) {
        elements_ = env_->GetFloatArrayElements(static_cast<jfloatArray>(array_), nullptr);
    }
    
    template<>
    SafeArrayElements<jfloat>::~SafeArrayElements() {
        if (elements_) {
            env_->ReleaseFloatArrayElements(static_cast<jfloatArray>(array_), elements_, mode_);
        }
    }
    
    // SafeString implementation
    SafeString::SafeString(JNIEnv* env, jstring jstr) : env_(env), jstr_(jstr), str_(nullptr) {
        if (jstr_) {
            str_ = env_->GetStringUTFChars(jstr_, nullptr);
        }
    }
    
    SafeString::~SafeString() {
        if (str_ && jstr_) {
            env_->ReleaseStringUTFChars(jstr_, str_);
        }
    }
}