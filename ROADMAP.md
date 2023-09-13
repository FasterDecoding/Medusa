# Roadmap

## Functionality
- [ ] Fine-tune Medusa heads together with LM head from scratch
- [ ] Distill from any model without access to the original training data
- [ ] Batched inference
- [ ] Fine-grained KV cache management

## Integration
### Local Deployment
- [ ] [mlc-llm](https://github.com/mlc-ai/mlc-llm)
- [ ] [exllama](https://github.com/turboderp/exllama)
- [ ] [llama.cpp](https://github.com/ggerganov/llama.cpp)
### Serving
- [ ] [vllm](https://github.com/vllm-project/vllm)
- [ ] [TGI](https://github.com/huggingface/text-generation-inference)
- [ ] [lightllm](https://github.com/ModelTC/lightllm)

## Research
- [ ] Optimize the tree-based attention to reduce additional computation
- [ ] Improve the acceptance scheme to generate more diverse sequences
