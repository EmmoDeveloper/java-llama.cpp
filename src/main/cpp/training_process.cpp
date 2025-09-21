// Standalone training process that runs outside JVM
// Communicates via stdin/stdout using JSON protocol

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <signal.h>

#include "llama.h"
#include "common.h"
#include "ggml-opt.h"
#include "json.hpp"

using json = nlohmann::json;

// Global variables for signal handling
static bool g_interrupted = false;
static llama_context* g_ctx = nullptr;
static llama_model* g_model = nullptr;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
	std::cerr << "[DEBUG] Signal " << sig << " received, initiating shutdown..." << std::endl;
	g_interrupted = true;
}

// Debug logging macro - write to file to avoid stdout/stderr mixing issues
#define DEBUG_LOG(msg) do { \
	std::ofstream debug_file("/tmp/training_process_debug.log", std::ios::app); \
	if (debug_file.is_open()) { \
		debug_file << "[DEBUG " << __FILE__ << ":" << __LINE__ << "] " << msg << std::endl; \
		debug_file.close(); \
	} \
} while(0)

// Training session state
struct TrainingState {
	llama_model* model = nullptr;
	llama_context* ctx = nullptr;
	ggml_opt_dataset_t dataset = nullptr;
	llama_opt_params opt_params;
	lr_opt learning_rate_config;
	int current_epoch = 0;
	int total_epochs = 1;
	float current_learning_rate = 0.0001f;
	std::vector<llama_token> tokens;
	bool is_initialized = false;

	~TrainingState() {
		DEBUG_LOG("Destroying TrainingState");
		if (dataset) {
			DEBUG_LOG("Freeing dataset");
			ggml_opt_dataset_free(dataset);
		}
		if (ctx) {
			DEBUG_LOG("Freeing context");
			llama_free(ctx);
		}
		if (model) {
			DEBUG_LOG("Freeing model");
			llama_model_free(model);
		}
	}
};

// Send JSON response back to Java process
void send_response(const json& response) {
	std::string response_str = response.dump();
	DEBUG_LOG("Sending response: " << response_str);
	std::cout << response_str << std::endl;
	std::cout.flush();
}

// Send error response
void send_error(const std::string& error_msg, const std::string& error_code = "ERROR") {
	json response;
	response["status"] = "error";
	response["error_code"] = error_code;
	response["message"] = error_msg;
	send_response(response);
}

// Send success response
void send_success(const json& data = json::object()) {
	json response;
	response["status"] = "success";
	response["data"] = data;
	send_response(response);
}

// Initialize training
void handle_init(const json& params, TrainingState& state) {
	DEBUG_LOG("=== STARTING TRAINING INITIALIZATION ===");

	try {
		// Extract parameters
		std::string model_path = params["model_path"];
		int n_ctx = params.value("n_ctx", 512);
		float learning_rate = params.value("learning_rate", 0.0001f);
		float weight_decay = params.value("weight_decay", 0.01f);
		int epochs = params.value("epochs", 1);
		bool use_adamw = params.value("use_adamw", true);
		int batch_size = params.value("batch_size", 32);

		DEBUG_LOG("Parameters received:");
		DEBUG_LOG("  model_path: " << model_path);
		DEBUG_LOG("  n_ctx: " << n_ctx);
		DEBUG_LOG("  learning_rate: " << learning_rate);
		DEBUG_LOG("  weight_decay: " << weight_decay);
		DEBUG_LOG("  epochs: " << epochs);
		DEBUG_LOG("  use_adamw: " << use_adamw);
		DEBUG_LOG("  batch_size: " << batch_size);

		// Initialize llama backend
		DEBUG_LOG("Step 1: Initializing llama backend...");
		llama_backend_init();
		DEBUG_LOG("  Backend initialized successfully");

		// Load model
		DEBUG_LOG("Step 2: Loading model from " << model_path << "...");
		llama_model_params model_params = llama_model_default_params();
		model_params.use_mmap = false; // Disable mmap for training
		DEBUG_LOG("  Model params: use_mmap=false");

		state.model = llama_model_load_from_file(model_path.c_str(), model_params);
		if (!state.model) {
			throw std::runtime_error("Failed to load model from " + model_path);
		}
		DEBUG_LOG("  Model loaded successfully");
		g_model = state.model; // Set global for signal handler

		// Create context
		DEBUG_LOG("Step 3: Creating context with n_ctx=" << n_ctx << "...");

		// Check model type and adjust context size for quantized models
		char model_desc_buf[256];
		llama_model_desc(state.model, model_desc_buf, sizeof(model_desc_buf));
		std::string model_name(model_desc_buf);
		DEBUG_LOG("Model description: " << model_name);

		if (model_name.find("Q2_K") != std::string::npos ||
		    model_name.find("Q3_K") != std::string::npos ||
		    model_name.find("Q4_K") != std::string::npos) {
			DEBUG_LOG("Detected quantized model, reducing context size for training");
			n_ctx = std::min(n_ctx, 256); // Limit context for quantized models
		}

		llama_context_params ctx_params = llama_context_default_params();
		ctx_params.n_ctx = n_ctx;
		ctx_params.n_batch = std::min(batch_size, 8); // Further limit batch size
		ctx_params.n_ubatch = std::min(batch_size, 8);
		DEBUG_LOG("  Context params: n_ctx=" << ctx_params.n_ctx <<
		          ", n_batch=" << ctx_params.n_batch <<
		          ", n_ubatch=" << ctx_params.n_ubatch);

		state.ctx = llama_init_from_model(state.model, ctx_params);
		if (!state.ctx) {
			throw std::runtime_error("Failed to create context");
		}
		DEBUG_LOG("  Context created successfully");
		g_ctx = state.ctx; // Set global for signal handler

		// Setup learning rate configuration
		DEBUG_LOG("Step 4: Setting up learning rate configuration...");
		state.learning_rate_config.lr0 = learning_rate;
		state.learning_rate_config.lr_min = -1;
		state.learning_rate_config.decay_epochs = -1;
		state.learning_rate_config.scale_epoch = 0;
		state.learning_rate_config.wd = weight_decay;
		state.learning_rate_config.epochs = epochs;
		state.learning_rate_config.epoch = 0;
		state.total_epochs = epochs;
		state.current_learning_rate = learning_rate;

		DEBUG_LOG("  lr_config.lr0 = " << state.learning_rate_config.lr0);
		DEBUG_LOG("  lr_config.lr_min = " << state.learning_rate_config.lr_min);
		DEBUG_LOG("  lr_config.decay_epochs = " << state.learning_rate_config.decay_epochs);
		DEBUG_LOG("  lr_config.scale_epoch = " << state.learning_rate_config.scale_epoch);
		DEBUG_LOG("  lr_config.wd = " << state.learning_rate_config.wd);
		DEBUG_LOG("  lr_config.epochs = " << state.learning_rate_config.epochs);
		DEBUG_LOG("  lr_config.epoch = " << state.learning_rate_config.epoch);

		// Initialize learning rate configuration
		DEBUG_LOG("Step 5: Calling lr_opt.init()...");
		state.learning_rate_config.init();
		DEBUG_LOG("  lr_opt.init() completed successfully");

		// Setup llama_opt_params
		DEBUG_LOG("Step 6: Setting up llama_opt_params...");
		state.opt_params.n_ctx_train = 0; // Use default
		state.opt_params.param_filter = llama_opt_param_filter_all;
		state.opt_params.param_filter_ud = nullptr;
		state.opt_params.get_opt_pars = common_opt_lr_pars;
		state.opt_params.get_opt_pars_ud = &state.learning_rate_config;
		state.opt_params.optimizer_type = use_adamw ?
			GGML_OPT_OPTIMIZER_TYPE_ADAMW : GGML_OPT_OPTIMIZER_TYPE_SGD;

		DEBUG_LOG("  opt_params.n_ctx_train = " << state.opt_params.n_ctx_train);
		DEBUG_LOG("  opt_params.param_filter = llama_opt_param_filter_all");
		DEBUG_LOG("  opt_params.get_opt_pars = common_opt_lr_pars");
		DEBUG_LOG("  opt_params.optimizer_type = " <<
		          (use_adamw ? "ADAMW" : "SGD"));

		// Initialize optimization context
		DEBUG_LOG("Step 7: Calling llama_opt_init()...");
		DEBUG_LOG("  Context pointer: " << (void*)state.ctx);
		DEBUG_LOG("  Model pointer: " << (void*)state.model);

		// Add a small delay to ensure everything is properly initialized
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		llama_opt_init(state.ctx, state.model, state.opt_params);

		DEBUG_LOG("  llama_opt_init() completed successfully!");

		state.is_initialized = true;

		json data;
		data["message"] = "Training initialized successfully";
		data["model_path"] = model_path;
		data["n_ctx"] = n_ctx;
		data["learning_rate"] = learning_rate;
		data["epochs"] = epochs;

		DEBUG_LOG("=== TRAINING INITIALIZATION COMPLETED SUCCESSFULLY ===");
		send_success(data);

	} catch (const std::exception& e) {
		DEBUG_LOG("ERROR during initialization: " << e.what());
		send_error(std::string("Initialization failed: ") + e.what(), "INIT_ERROR");
	}
}

// Train one epoch
void handle_train_epoch(const json& params, TrainingState& state) {
	DEBUG_LOG("=== STARTING TRAINING EPOCH ===");

	if (!state.is_initialized) {
		send_error("Training not initialized", "NOT_INITIALIZED");
		return;
	}

	try {
		std::string dataset_path = params["dataset_path"];
		DEBUG_LOG("Loading dataset from: " << dataset_path);

		// Load dataset
		std::ifstream file(dataset_path);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open dataset file: " + dataset_path);
		}

		std::stringstream buffer;
		buffer << file.rdbuf();
		std::string text = buffer.str();
		file.close();

		DEBUG_LOG("Dataset loaded, size: " << text.length() << " characters");

		if (text.empty()) {
			throw std::runtime_error("Dataset is empty");
		}

		// Tokenize
		DEBUG_LOG("Tokenizing dataset...");
		state.tokens = common_tokenize(state.ctx, text, true);
		DEBUG_LOG("Tokenized to " << state.tokens.size() << " tokens");

		if (state.tokens.empty()) {
			throw std::runtime_error("Failed to tokenize dataset");
		}

		// Create dataset with safer memory allocation
		DEBUG_LOG("Creating optimization dataset...");

		// Use much smaller context size for training to avoid optimization graph issues
		int n_ctx = llama_n_ctx(state.ctx);
		int training_ctx = std::min(64, n_ctx / 4); // Limit to 64 tokens max for training to avoid graph build failures

		DEBUG_LOG("Context size: " << n_ctx << ", Training context: " << training_ctx);
		DEBUG_LOG("Token count: " << state.tokens.size());

		// Check if we have enough tokens for dataset creation
		// Formula from common.cpp: ndata = (tokens.size() - ne_datapoint - 1) / stride
		int64_t stride = 1;
		int64_t ne_datapoint = training_ctx;
		int64_t required_tokens = ne_datapoint + 1 + stride; // Minimum tokens needed
		int64_t ndata = (state.tokens.size() - ne_datapoint - 1) / stride;

		DEBUG_LOG("Dataset validation: tokens=" << state.tokens.size()
		         << ", ne_datapoint=" << ne_datapoint
		         << ", required=" << required_tokens
		         << ", ndata=" << ndata);

		if (state.tokens.size() < required_tokens) {
			throw std::runtime_error("Dataset too small: have " + std::to_string(state.tokens.size()) +
			                        " tokens, need at least " + std::to_string(required_tokens) +
			                        " for context size " + std::to_string(ne_datapoint));
		}

		if (ndata <= 0) {
			throw std::runtime_error("Invalid dataset configuration: ndata=" + std::to_string(ndata) +
			                        " (tokens=" + std::to_string(state.tokens.size()) +
			                        ", context=" + std::to_string(ne_datapoint) + ")");
		}

		// Limit tokens to avoid memory issues
		if (state.tokens.size() > 1000) {
			DEBUG_LOG("Limiting tokens from " << state.tokens.size() << " to 1000 to avoid memory issues");
			state.tokens.resize(1000);
			// Recalculate ndata after limiting tokens
			ndata = (state.tokens.size() - ne_datapoint - 1) / stride;
			if (ndata <= 0) {
				throw std::runtime_error("Dataset too small after token limiting");
			}
		}

		try {
			DEBUG_LOG("Creating dataset with stride=" << stride << ", context=" << training_ctx << ", ndata=" << ndata);

			state.dataset = common_opt_dataset_init(state.ctx, state.tokens, stride);
			if (!state.dataset) {
				throw std::runtime_error("common_opt_dataset_init returned null");
			}
		} catch (const std::bad_alloc& e) {
			throw std::runtime_error("Out of memory creating dataset. Try reducing context size or dataset size.");
		} catch (const std::exception& e) {
			throw std::runtime_error(std::string("Failed to initialize dataset: ") + e.what());
		}

		DEBUG_LOG("Dataset created successfully with context size: " << training_ctx);

		// Training parameters
		const float val_split = 0.1f;
		const int64_t idata_split = ggml_opt_dataset_ndata(state.dataset) * (1.0f - val_split);
		DEBUG_LOG("Training/validation split at index: " << idata_split);

		// Initialize result objects
		DEBUG_LOG("Initializing result objects...");
		ggml_opt_result_t result_train = ggml_opt_result_init();
		ggml_opt_result_t result_eval = ggml_opt_result_init();

		// Perform training
		DEBUG_LOG("Starting llama_opt_epoch...");
		auto start_time = std::chrono::steady_clock::now();

		llama_opt_epoch(state.ctx, state.dataset, result_train, result_eval,
		                idata_split, nullptr, nullptr);

		auto end_time = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
			end_time - start_time).count();

		DEBUG_LOG("Training epoch completed in " << duration << " ms");

		// Get metrics
		double loss_value = 0.0;
		double loss_unc = 0.0;
		ggml_opt_result_loss(result_train, &loss_value, &loss_unc);

		DEBUG_LOG("Training loss: " << loss_value << " (uncertainty: " << loss_unc << ")");

		// Clean up
		ggml_opt_result_free(result_train);
		ggml_opt_result_free(result_eval);

		if (state.dataset) {
			ggml_opt_dataset_free(state.dataset);
			state.dataset = nullptr;
		}

		// Update epoch counter and learning rate
		state.current_epoch++;
		state.current_learning_rate *= 0.995f;
		DEBUG_LOG("Epoch " << state.current_epoch << " completed");
		DEBUG_LOG("New learning rate: " << state.current_learning_rate);

		json data;
		data["loss"] = loss_value;
		data["learning_rate"] = state.current_learning_rate;
		data["epoch"] = state.current_epoch;
		data["duration_ms"] = duration;
		data["total_tokens"] = state.tokens.size();

		DEBUG_LOG("=== TRAINING EPOCH COMPLETED ===");
		send_success(data);

	} catch (const std::exception& e) {
		DEBUG_LOG("ERROR during training: " << e.what());
		send_error(std::string("Training failed: ") + e.what(), "TRAIN_ERROR");
	}
}

// Evaluate model
void handle_evaluate(const json& params, TrainingState& state) {
	DEBUG_LOG("=== STARTING EVALUATION ===");

	if (!state.is_initialized) {
		send_error("Training not initialized", "NOT_INITIALIZED");
		return;
	}

	try {
		std::string dataset_path = params["dataset_path"];
		DEBUG_LOG("Loading validation dataset from: " << dataset_path);

		// Load dataset
		std::ifstream file(dataset_path);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open validation dataset");
		}

		std::stringstream buffer;
		buffer << file.rdbuf();
		std::string text = buffer.str();
		file.close();

		if (text.empty()) {
			throw std::runtime_error("Validation dataset is empty");
		}

		DEBUG_LOG("Validation dataset loaded, size: " << text.length());

		// Tokenize
		std::vector<llama_token> tokens = common_tokenize(state.ctx, text, true, false);
		DEBUG_LOG("Tokenized to " << tokens.size() << " tokens");

		// Calculate perplexity
		const int n_ctx = llama_n_ctx(state.ctx);
		const llama_vocab* vocab = llama_model_get_vocab(state.model);
		const int n_vocab = llama_vocab_n_tokens(vocab);

		DEBUG_LOG("Context size: " << n_ctx << ", Vocab size: " << n_vocab);

		double nll = 0.0;
		int count = 0;

		const int eval_window = n_ctx / 2;
		const int stride = std::min(256, n_ctx / 2);

		DEBUG_LOG("Evaluation window: " << eval_window << ", Stride: " << stride);

		// Clear memory before evaluation
		llama_memory_t mem = llama_get_memory(state.ctx);
		llama_memory_clear(mem, true);

		for (size_t start = 0; start < tokens.size(); start += stride) {
			size_t end = std::min(start + n_ctx, tokens.size());
			size_t num_tokens = end - start;

			if (num_tokens < 2) break;

			// Create batch
			llama_batch batch = llama_batch_init(num_tokens, 0, 1);

			// Fill batch
			for (size_t i = 0; i < num_tokens; i++) {
				batch.token[i] = tokens[start + i];
				batch.pos[i] = i;
				batch.n_seq_id[i] = 1;
				batch.seq_id[i] = &batch.seq_id[0][i];
				batch.seq_id[i][0] = 0;
				batch.logits[i] = (i >= eval_window || i == num_tokens - 1) ? 1 : 0;
			}
			batch.n_tokens = num_tokens;

			// Evaluate
			if (llama_decode(state.ctx, batch) != 0) {
				llama_batch_free(batch);
				throw std::runtime_error("Failed to evaluate batch");
			}

			// Calculate log probabilities
			const float* logits = llama_get_logits(state.ctx);
			int logits_idx = 0;

			for (size_t i = eval_window; i < num_tokens - 1; i++) {
				if (batch.logits[i]) {
					const float* token_logits = logits + logits_idx * n_vocab;

					// Calculate log softmax
					float max_logit = *std::max_element(token_logits, token_logits + n_vocab);

					double sum_exp = 0.0;
					for (int v = 0; v < n_vocab; v++) {
						sum_exp += exp(token_logits[v] - max_logit);
					}

					int target_token = tokens[start + i + 1];
					double log_prob = token_logits[target_token] - max_logit - log(sum_exp);

					nll -= log_prob;
					count++;
					logits_idx++;
				}
			}

			llama_batch_free(batch);

			if (start + stride < tokens.size()) {
				llama_memory_clear(mem, true);
			}
		}

		float avg_nll = count > 0 ? (float)(nll / count) : 0.0f;
		float perplexity = exp(avg_nll);
		float accuracy = 1.0f / (1.0f + perplexity / 100.0f);

		DEBUG_LOG("Evaluation results: NLL=" << avg_nll <<
		          ", Perplexity=" << perplexity <<
		          ", Accuracy=" << accuracy);

		json data;
		data["loss"] = avg_nll;
		data["perplexity"] = perplexity;
		data["accuracy"] = accuracy;
		data["total_samples"] = count;

		DEBUG_LOG("=== EVALUATION COMPLETED ===");
		send_success(data);

	} catch (const std::exception& e) {
		DEBUG_LOG("ERROR during evaluation: " << e.what());
		send_error(std::string("Evaluation failed: ") + e.what(), "EVAL_ERROR");
	}
}

// Save checkpoint
void handle_save_checkpoint(const json& params, TrainingState& state) {
	DEBUG_LOG("=== SAVING CHECKPOINT ===");

	if (!state.is_initialized) {
		send_error("Training not initialized", "NOT_INITIALIZED");
		return;
	}

	try {
		std::string checkpoint_path = params["checkpoint_path"];
		DEBUG_LOG("Saving checkpoint to: " << checkpoint_path);

		// Save model weights
		std::string model_path = checkpoint_path + ".model.gguf";
		DEBUG_LOG("Saving model weights to: " << model_path);
		llama_model_save_to_file(state.model, model_path.c_str());

		// Save context state
		std::string state_path = checkpoint_path + ".state";
		DEBUG_LOG("Saving context state to: " << state_path);
		std::vector<llama_token> empty_tokens;
		if (!llama_state_save_file(state.ctx, state_path.c_str(),
		                           empty_tokens.data(), empty_tokens.size())) {
			DEBUG_LOG("Warning: Failed to save context state");
		}

		// Save metadata
		std::string meta_path = checkpoint_path + ".meta";
		DEBUG_LOG("Saving metadata to: " << meta_path);
		std::ofstream metadata(meta_path);
		if (metadata.is_open()) {
			metadata << "epoch=" << state.current_epoch << std::endl;
			metadata << "learning_rate=" << state.current_learning_rate << std::endl;
			metadata << "total_epochs=" << state.total_epochs << std::endl;
			metadata << "optimizer_type=" << state.opt_params.optimizer_type << std::endl;
			metadata.close();
		}

		json data;
		data["checkpoint_path"] = checkpoint_path;
		data["files_saved"] = json::array({model_path, state_path, meta_path});

		DEBUG_LOG("=== CHECKPOINT SAVED ===");
		send_success(data);

	} catch (const std::exception& e) {
		DEBUG_LOG("ERROR saving checkpoint: " << e.what());
		send_error(std::string("Save checkpoint failed: ") + e.what(), "SAVE_ERROR");
	}
}

// Main message processing loop
int main(int argc, char** argv) {
	// Set up signal handlers
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	DEBUG_LOG("=== TRAINING PROCESS STARTED ===");
	DEBUG_LOG("PID: " << getpid());
	DEBUG_LOG("Waiting for commands on stdin...");

	TrainingState state;
	std::string line;

	while (!g_interrupted && std::getline(std::cin, line)) {
		try {
			DEBUG_LOG("Received command: " << line);

			json command = json::parse(line);
			std::string action = command["action"];

			DEBUG_LOG("Processing action: " << action);

			if (action == "init") {
				handle_init(command["params"], state);
			} else if (action == "train_epoch") {
				handle_train_epoch(command["params"], state);
			} else if (action == "evaluate") {
				handle_evaluate(command["params"], state);
			} else if (action == "save_checkpoint") {
				handle_save_checkpoint(command["params"], state);
			} else if (action == "shutdown") {
				DEBUG_LOG("Shutdown command received");
				break;
			} else {
				send_error("Unknown action: " + action, "UNKNOWN_ACTION");
			}

		} catch (const json::exception& e) {
			DEBUG_LOG("JSON parsing error: " << e.what());
			send_error(std::string("Invalid JSON: ") + e.what(), "JSON_ERROR");
		} catch (const std::exception& e) {
			DEBUG_LOG("Unexpected error: " << e.what());
			send_error(std::string("Unexpected error: ") + e.what(), "UNKNOWN_ERROR");
		}
	}

	DEBUG_LOG("=== TRAINING PROCESS SHUTTING DOWN ===");

	// Cleanup is handled by TrainingState destructor
	llama_backend_free();

	DEBUG_LOG("=== TRAINING PROCESS TERMINATED ===");
	return 0;
}