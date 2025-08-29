#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include "llama.h"
#include "completion_task.h"

struct TaskRequest {
	int id;
	std::string prompt;
	int n_predict = 10;
};

struct TaskResult {
	int task_id;
	std::string text;
	bool is_final;
	bool is_error;
	std::string error_msg;
	
	TaskResult(int id, const std::string& t, bool final = false, bool error = false, const std::string& err = "") 
		: task_id(id), text(t), is_final(final), is_error(error), error_msg(err) {}
};

struct LlamaServer {
	llama_model* model = nullptr;
	llama_context* ctx = nullptr;
	llama_sampler* sampler = nullptr;
	bool embedding_mode = false;
	bool reranking_mode = false;
	
	// Task management
	std::queue<TaskRequest> task_queue;
	std::mutex task_queue_mutex;
	std::condition_variable task_queue_cv;
	
	// Result management
	std::unordered_map<int, std::queue<TaskResult>> task_results;
	std::mutex result_mutex;
	
	// Active tasks
	std::unordered_map<int, std::unique_ptr<CompletionTask>> active_tasks;
	std::mutex active_tasks_mutex;
	
	// Server thread
	std::thread server_thread;
	std::atomic<bool> should_stop{false};
	int next_task_id = 1;
	
	// Server main loop
	void server_loop();
	void process_task(const TaskRequest& request);
	void generate_tokens(CompletionTask* task);
	
	void start_server() {
		server_thread = std::thread(&LlamaServer::server_loop, this);
	}
	
	void stop_server() {
		should_stop = true;
		task_queue_cv.notify_all();
		if (server_thread.joinable()) {
			server_thread.join();
		}
	}
	
	~LlamaServer() {
		stop_server();
		if (sampler) llama_sampler_free(sampler);
		if (ctx) llama_free(ctx);
		if (model) llama_model_free(model);
	}
};