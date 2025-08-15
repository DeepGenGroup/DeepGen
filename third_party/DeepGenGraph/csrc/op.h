#include <torch/extension.h>
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, float epsilon);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void rotary_embedding_online(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, int head_size, float rope_theta);