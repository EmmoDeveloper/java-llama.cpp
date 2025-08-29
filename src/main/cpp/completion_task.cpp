#include "completion_task.h"

CompletionTask::CompletionTask(int task_id, const std::string& p, int predict, const std::string& g) 
	: id(task_id), prompt(p), grammar(g), n_predict(predict) {
}

CompletionTask::~CompletionTask() {
	if (task_sampler) {
		llama_sampler_free(task_sampler);
		task_sampler = nullptr;
	}
}