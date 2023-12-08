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

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

api_base_pool = []

# List models API
for i in range(10):
    openai.api_base = "http://localhost:800{}/v1".format(i)
    try:     
        models = openai.Model.list()["data"][0]["id"]
        print(openai.api_base, models)
        api_base_pool.append(openai.api_base)
    except:
        break

print("API base pool: ", api_base_pool)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--num_threads", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--chat", action="store_true")
args = parser.parse_args()

# Assuming the ShareGPT format
data = json.load(open(args.data_path, "r"))

def generate_data(messages, idx):
    try:
        # load balanced
        openai.api_base = api_base_pool[idx % len(api_base_pool)]
        model_name=openai.Model.list()["data"][0]["id"]

        if args.chat:
            converted_messages = []
            output_messages = []
            if messages[0]["from"] == "system":
                converted_messages.append(
                    {
                        "role": "system",
                        "content": messages[0]["text"],
                    }
                )
                output_messages.append(messages[0])
                messages = messages[1:]
            for message in messages[::2]:
                if message["from"] != "human":
                    return
                converted_messages.append(
                    {
                        "role": "user",
                        "content": message["value"],
                    }
                )
                try:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=converted_messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    if response.choices[0]['finish_reason'] == "length":
                        break
                    response = response.choices[0]['message']['content'].strip()
                    output_messages.append(message)
                    output_messages.append(
                        {
                            "from": "gpt",
                            "value": response,
                        }
                    )
                    converted_messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    )
                except:
                    break
            if len(output_messages) == 0:
                return
            with open(args.output_path, "a") as f:
                # write in share gpt format
                f.write(json.dumps({"conversations": output_messages}) + "\n")
        else:
            conv = get_conversation_template(model_name)
            if messages[0]["from"] == "system":
                conv.system_message = messages[0]["text"]
                messages = messages[1:]
            conv.append_message(conv.roles[0], messages[0]["value"])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            response = openai.Completion.create(
                model=model_name,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                ignore_eos=True,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            response = response.choices[0]['text'].strip()
            with open(args.output_path, "a") as f:
                # write in share gpt format
                f.write(json.dumps({"text": prompt+response}) + "\n")
    except Exception as e:
        print(e)
        print(prompt)
        print("Failed to generate data")

# if output_path exists, count the number of lines and skip the first n data
start = 0
if os.path.exists(args.output_path):
    with open(args.output_path, "r") as f:
        start = len(f.readlines())
        print("Skip first {} data".format(start))

with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for idx, sample in enumerate(data[start:]):
            future = executor.submit(
                generate_data,
                sample["conversations"],
                idx,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()