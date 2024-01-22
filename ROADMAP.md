# Roadmap

## Functionality
- [ ] Batched inference
- [ ] Fine-grained KV cache management
- [x] Explore tree sparsity
- [x] Fine-tune Medusa heads together with LM head from scratch
- [x] Distill from any model without access to the original training data

## Integration
### Local Deployment
- [ ] [mlc-llm](https://github.com/mlc-ai/mlc-llm)
- [ ] [exllama](https://github.com/turboderp/exllama)
- [ ] [llama.cpp](https://github.com/ggerganov/llama.cpp)
### Serving
- [ ] [vllm](https://github.com/vllm-project/vllm)
- [ ] [lightllm](https://github.com/ModelTC/lightllm)
- [x] [TGI](https://github.com/huggingface/text-generation-inference)
- [x] [TensorRT](https://github.com/NVIDIA/TensorRT-LLM)