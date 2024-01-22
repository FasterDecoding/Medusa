
We use [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/0cd24d711fe90d0c1aae5bde03fe98ee48ae52f8/alpaca_eval.json) dataset for evaluating each head's accuracy during generation in `heads_accuracy.py`.

```
python heads_accuracy.py --model_path 'FasterDecoding/medusa-vicuna-7b-v1.3' --model_name 'medusa-vicuna-7b-v1.3' --medusa_num_heads 5 --data_path '../../data/alpaca_eval.json'
```


To create the tree and plot the tree (requires `pygraphviz` package), please run:

```
python gen_results.py --accuracy-path '../../data/medusa-vicuna-7b-v1.3_heads_accuracy.pt' --output-path '../../data/graph.jpg'
```

If you want to use the tree, please add the generated tree (in a nested tuple) to `../model/medusa_choices.py`.

Citation:

```
@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}```