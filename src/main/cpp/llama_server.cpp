#include "llama_server.h"
#include "sampling.h"
#include <iostream>

void LlamaServer::server_loop() {
	while (!should_stop) {
		std::unique_lock<std::mutex> lock(task_queue_mutex);
		task_queue_cv.wait(lock, [this] { return !task_queue.empty() || should_stop; });
		
		if (should_stop) break;
		
		if (!task_queue.empty()) {
			TaskRequest request = task_queue.front();
			task_queue.pop();
			lock.unlock();
			
			process_task(request);
		}
	}
}

void LlamaServer::process_task(const TaskRequest& request) {
	auto task = std::make_unique<CompletionTask>(request.id, request.prompt, request.n_predict);
	
	// Tokenize prompt
	const llama_vocab* vocab = llama_model_get_vocab(model);
	std::vector<llama_token> tokens;
	tokens.resize(request.prompt.length() + 1);
	
	int n_tokens = llama_tokenize(vocab, request.prompt.c_str(), request.prompt.length(), 
								  tokens.data(), tokens.size(), true, false);
	
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		n_tokens = llama_tokenize(vocab, request.prompt.c_str(), request.prompt.length(),
								  tokens.data(), tokens.size(), true, false);
	}
	
	if (n_tokens < 0) {
		std::lock_guard<std::mutex> result_lock(result_mutex);
		task_results[request.id].emplace(request.id, "", true, true, "Tokenization failed");
		return;
	}
	
	tokens.resize(n_tokens);
	task->prompt_tokens = tokens;
	task->state = TASK_STATE_PROCESSING_PROMPT;
	
	// Process prompt
	llama_batch batch = llama_batch_init(n_tokens, 0, 1);
	for (int i = 0; i < n_tokens; i++) {
		batch.token[i] = tokens[i];
		batch.pos[i] = i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = 0;
		batch.logits[i] = (i == n_tokens - 1); // Only compute logits for last token
	}
	batch.n_tokens = n_tokens;
	
	if (llama_decode(ctx, batch) != 0) {
		llama_batch_free(batch);
		std::lock_guard<std::mutex> result_lock(result_mutex);
		task_results[request.id].emplace(request.id, "", true, true, "Prompt processing failed");
		return;
	}
	
	llama_batch_free(batch);
	task->current_pos = n_tokens;
	task->state = TASK_STATE_GENERATING;
	
	// Store task
	{
		std::lock_guard<std::mutex> tasks_lock(active_tasks_mutex);
		active_tasks[request.id] = std::move(task);
	}
	
	// Generate tokens
	generate_tokens(active_tasks[request.id].get());
}

void LlamaServer::generate_tokens(CompletionTask* task) {
	for (int i = 0; i < task->n_predict && !task->cancelled; i++) {
		// Sample next token
		llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
		
		// Check for end of generation
		const llama_vocab* vocab = llama_model_get_vocab(model);
		if (llama_vocab_is_eog(vocab, new_token)) {
			std::lock_guard<std::mutex> result_lock(result_mutex);
			task_results[task->id].emplace(task->id, task->current_text, true);
			task->state = TASK_STATE_COMPLETED;
			return;
		}
		
		// Convert token to text
		char piece[256];
		int piece_len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
		if (piece_len > 0) {
			task->current_text.append(piece, piece_len);
		}
		
		// Add result to queue
		{
			std::lock_guard<std::mutex> result_lock(result_mutex);
			task_results[task->id].emplace(task->id, task->current_text, false);
		}
		
		// Process next token
		llama_batch batch = llama_batch_init(1, 0, 1);
		batch.token[0] = new_token;
		batch.pos[0] = task->current_pos++;
		batch.n_seq_id[0] = 1;
		batch.seq_id[0][0] = 0;
		batch.logits[0] = true;
		batch.n_tokens = 1;
		
		if (llama_decode(ctx, batch) != 0) {
			llama_batch_free(batch);
			std::lock_guard<std::mutex> result_lock(result_mutex);
			task_results[task->id].emplace(task->id, task->current_text, true, true, "Token generation failed");
			task->state = TASK_STATE_COMPLETED;
			return;
		}
		
		llama_batch_free(batch);
	}
	
	// Generation completed
	std::lock_guard<std::mutex> result_lock(result_mutex);
	task_results[task->id].emplace(task->id, task->current_text, true);
	task->state = TASK_STATE_COMPLETED;
}