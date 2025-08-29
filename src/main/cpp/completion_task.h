#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include "llama.h"

enum TaskState {
    TASK_STATE_PENDING,
    TASK_STATE_PROCESSING_PROMPT,
    TASK_STATE_GENERATING,
    TASK_STATE_COMPLETED,
    TASK_STATE_CANCELLED
};

class CompletionTask {
public:
    int id;
    std::string prompt;
    std::string grammar;  // Grammar string for constrained generation
    llama_sampler* task_sampler = nullptr;  // Task-specific sampler (with grammar if provided)
    TaskState state = TASK_STATE_PENDING;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> generated_tokens;
    std::string current_text;
    int n_predict = 10;
    int current_pos = 0;
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    
    CompletionTask(int task_id, const std::string& p, int predict = 10, const std::string& g = "");
    ~CompletionTask();
};