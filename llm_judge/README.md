# LLM Judge
| [Original Github Repository](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

## Installation

| [Guide](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)

## Usage

We report the 3 times running results of the Medusa X Vicuna v1.3 7/13/33b on a single A100 in `./data/mt_bench/model_answer/`. The original settings are: `temperature` (it is deprecated and use the default LLM Judge setting), `posterior_threshold=0.09`, `posterior_alpha=0.3`.

- Run benchmark


```
export CUDA_VISIBLE_DEVICES=0 # set the GPU id
python gen_model_answer_medusa.py  --model-path FasterDecoding/medusa-vicuna-7b-v1.3 --model-id medusa-vicuna-7b-v1.3-0
python gen_model_answer_medusa.py  --model-path FasterDecoding/medusa-vicuna-13b-v1.3 --model-id medusa-vicuna-13b-v1.3-0
python gen_model_answer_medusa.py  --model-path FasterDecoding/medusa-vicuna-33b-v1.3 --model-id medusa-vicuna-33b-v1.3-0
```

- Run baseline: replace `gen_model_answer_medusa.py` with `gen_model_answer_baseline.py` (Please note we only implement the greedy inference for wall-time comparison. If you want to use the sampling generator, please refer to the original repository.)


- Query the results

```
export OPENAI_API_KEY=$OPENAI_API_KEYs # set the OpenAI API key
python gen_judgement.py --model-list medusa-vicuna-7b-v1.3-0-temperature-0.0-posterior_threshold-0.09-posterior_alpha-0.3 
```

- Show results

To obtain the results of GPT-4 judge for Vicuna-7b ( Huggingface greedy | Huggingface sampling | Medusa sampling), run:

```
python show_result.py
```

## Citation
Please cite the original paper if you find the code or datasets helpful.
```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena}, 
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```