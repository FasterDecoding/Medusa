import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm

import argparse
import random

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoTokenizer

# Use the same arguments as in generate.py
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta")
args = parser.parse_args()

conv = get_conversation_template(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

data = []
with open(args.input_path) as f:
    for line in f.readlines():
        data.append(json.loads(line))

def convert(text):
    messages = []

    for turn in text.split(conv.roles[0]):
        pairs = turn.split(conv.roles[1])
        if len(pairs) != 2:
            continue
        messages.append({
            "from": "human",
            "value": pairs[0].split(conv.sep)[0].strip()
        })
        messages.append({
            "from": "gpt",
            "value": pairs[1].split(conv.sep)[0].strip()
        })
    # pop the last message because it might be incomplete
    if len(messages) > 0:
        messages.pop()
    # make sure number of messages is even
    if len(messages) % 2 == 1:
        messages.pop()
    return {"conversations": messages}

sharegpt_data = []
for d in tqdm.tqdm(data):
    sample = convert(d["text"])
    if len(sample["conversations"]) < 2:
        continue
    sharegpt_data.append(sample)

# dump to jsonl
with open(args.input_path.replace(".jsonl", "_sharegpt.jsonl"), "w") as f:
    for d in sharegpt_data:
        f.write(json.dumps(d) + "\n")