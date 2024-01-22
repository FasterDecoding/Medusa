"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template

# Medusa imports
import transformers


from medusa.model.utils import *
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import initialize_past_key_values
from medusa.model.medusa_choices import *

def medusa_forward(input_ids, model, tokenizer, medusa_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    # Cache medusa buffers (the fixed patterns for tree attention)
    if hasattr(model, "medusa_choices") and model.medusa_choices == medusa_choices:
        # Load the cached medusa buffer
        medusa_buffers = model.medusa_buffers
    else:
        # Initialize the medusa buffer
        medusa_buffers = generate_medusa_buffers(
            medusa_choices, device=model.base_model.device
        )
    model.medusa_buffers = medusa_buffers
    model.medusa_choices = medusa_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_medusa_mode(model)
    # medusa_logits, logits = initialize_medusa(
    #         input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
    # )
    outputs = model.base_model(input_ids, past_key_values = past_key_values, use_cache=True)
    new_token = 0
    
    torch.cuda.synchronize()
    start_time = time.time()

    for idx in range(max_steps): 
        # candidates, tree_candidates = generate_candidates(
        #         medusa_logits,
        #         logits,
        #         medusa_buffers["tree_indices"],
        #         medusa_buffers["retrieve_indices"],
        #     )
        # medusa_logits, logits, outputs = tree_decoding(
        #         model,
        #         tree_candidates,
        #         past_key_values,
        #         medusa_buffers["medusa_position_ids"],
        #         input_ids,
        #         medusa_buffers["retrieve_indices"],
        #     )
        # best_candidate, accept_length = evaluate_posterior(
        #         logits, candidates, temperature, posterior_threshold, posterior_alpha
        #     )
        # input_ids, logits, medusa_logits, new_token = update_inference_inputs(
        #         input_ids,
        #         candidates,
        #         best_candidate,
        #         accept_length,
        #         medusa_buffers["retrieve_indices"],
        #         outputs,
        #         logits,
        #         medusa_logits,
        #         new_token,
        #         past_key_values_data,
        #         current_length_data,
        #     )
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model.base_model(input_id, use_cache=True, past_key_values = past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    return input_ids, new_token, idx, total_time

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    medusa_choices,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                posterior_threshold,
                posterior_alpha,
                medusa_choices,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    medusa_choices,
):
    
    # Medusa model setup
    num_heads = 4

    model = MedusaModel.from_pretrained(
        model_path,
        medusa_num_heads = num_heads,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:',model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    
    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = get_conversation_template(model_id)
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # if temperature < 1e-4:
            #     do_sample = False
            # else:
            #     do_sample = True

            # some models may error out when generating long outputs
            try:
                # torch.cuda.synchronize()
                # start_time = time.time()
                output_ids, new_token, idx, total_time = medusa_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    medusa_choices,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                )
                # torch.cuda.synchronize()
                # total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]) :]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')


    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                # if temperature < 1e-4:
                #     do_sample = False
                # else:
                #     do_sample = True

                # some models may error out when generating long outputs
                try:
                    # torch.cuda.synchronize()
                    # start_time = time.time()
                    output_ids, new_token, idx, total_time = medusa_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        medusa_choices,
                        temperature,
                        posterior_threshold,
                        posterior_alpha,
                    )
                    # torch.cuda.synchronize()
                    # total_time = time.time() - start_time
                    # if model.config.is_encoder_decoder:
                    #     output_ids = output_ids[0]
                    # else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    # YL: Medusa args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )

    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for medusa sampling.",
    )
    
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for medusa sampling.",
    )

    parser.add_argument(
        "--medusa-choices",
        type=str,
        default="mc_sim_7b_63",
        help="The medusa choices for medusa sampling.",
    )




    args = parser.parse_args()

    args.model_id = args.model_id+"-greedy-infonly"
    args.medusa_choices = eval(args.medusa_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,

        args.temperature,
        args.posterior_threshold,
        args.posterior_alpha,
        args.medusa_choices,
    )

    reorg_answer_file(answer_file)