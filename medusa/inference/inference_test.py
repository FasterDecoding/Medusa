# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
"""
Chat with a model with command line interface.

Usage:
python3 -m medusa.inference.cli --model <model_name_or_path>
Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os
import re
import sys
import torch
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import json
from medusa.model.medusa_model import MedusaModel
import pdb

def main(args):
    prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {0} ASSISTANT:"
    # prompt = ["你叫什么名字"]
    prompt = ["你叫什么名字", "中国的首都是哪里呢？"]
    prompt = [prefix.format(p) for p in prompt]
    model = MedusaModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = model.get_tokenizer()
    # 使用tokenizer处理批量输入
    encoded_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    # 将编码后的输入移动到模型所在的设备
    input_ids = encoded_inputs['input_ids'].to(model.base_model.device)
    attention_mask = encoded_inputs['attention_mask'].to(model.base_model.device) 
    for output in model.medusa_generate(
            input_ids,
            attention_mask=attention_mask,
            temperature=args.temperature,
            max_steps=args.max_steps,
        ):
        print(output['text'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
