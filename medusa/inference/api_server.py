import argparse
import json
from typing import AsyncGenerator
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from medusa.model.medusa_model import MedusaModel
import asyncio
from collections import deque
import uuid
from contextlib import asynccontextmanager

TIMEOUT_KEEP_ALIVE = 5  # seconds.
engine = None
max_batch_size = 5
request_queue = deque()
id2result = {}

async def handle_request(request_data):
    request_queue.append(request_data)

async def get_batch_from_queue():
    prompts = []
    ids = []
    if args.origin_model:
        request_dict_ = {"temperature":0.5, "max_tokens":150, "top_p": 0.85}
    else:
        request_dict_ = {"temperature":0.0, "max_tokens":150, "top_p": 0.85}
    max_tokens = None
    start_time = asyncio.get_event_loop().time()  # 获取当前时间
    while len(prompts) < max_batch_size:
        # 检查是否超时
        if asyncio.get_event_loop().time() - start_time >= 0.03:
            break
        # 如果队列为空，等待1ms再尝试
        if not request_queue:
            await asyncio.sleep(0.001)
            continue
        request_dict = request_queue.popleft()
        if request_dict.get("max_tokens", None):
            if max_tokens:
                max_tokens = max(max_tokens, request_dict["max_tokens"])
            else:
                max_tokens = request_dict["max_tokens"]
        prompts.append(request_dict.pop("prompt"))
        ids.append(request_dict.pop("unique_id"))
    if max_tokens:
        request_dict_["max_tokens"] = max_tokens
    if len(prompts) > 0 and request_dict.get("temperature", None):
        request_dict_["temperature"] = request_dict["temperature"]
    if len(prompts) > 0 and request_dict.get("top_p", None):
        request_dict_["top_p"] = request_dict["top_p"]    
    return prompts, ids, request_dict_


async def run_model():
    while True:
        prompt, ids, request_dict = await get_batch_from_queue()
        if len(prompt) >0:
            print(f"batch size: {len(prompt)}")
            encoded_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_inputs['input_ids'].to(engine.base_model.device)
            attention_mask = encoded_inputs['attention_mask'].to(engine.base_model.device)     
            for request_output in engine.medusa_generate(
                                                input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                temperature=request_dict["temperature"],
                                                max_steps=request_dict["max_tokens"],
                                                top_p=request_dict["top_p"]
                                            ):
                await asyncio.sleep(0.001)
                for index, id in enumerate(ids):
                    if id2result[id] is None:
                        id2result[id] = {'text':None, 'sign':None, 'finished':False}
                    if id2result[id]['text'] != request_output["text"][index]:
                        id2result[id]['text'] = request_output["text"][index] #full_sentences[index]
                        id2result[id]['sign'] = str(uuid.uuid4())
   
            for index, id in enumerate(ids):
                id2result[id]['finished'] = True
        else:
            pass

app = FastAPI()

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_model())

@app.post("/generate")
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    unique_id = str(uuid.uuid4())
    request_dict["unique_id"] = unique_id
    id2result[unique_id] = None
    await handle_request(request_dict) ##接收数据放入queue

    async def stream_results():
        previous_sign = None
        while True: ##循环取输出输出
            result = id2result.get(unique_id, None)
            if result is not None:
                if result['sign'] != previous_sign: ##是否更新
                    full_sentence = result['text']
                    ret = {"text":[full_sentence]}
                    previous_sign = result['sign']
                    yield (json.dumps(ret) + "\0").encode("utf-8")  
                else:
                    if result['finished']: ##是否写完
                        print(f"{unique_id} 全部输出完毕，删除")
                        id2result.pop(unique_id)
                        break
                    await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0.001)

    return StreamingResponse(stream_results()) ##返回数据    

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--origin-model", action="store_true")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")

    args = parser.parse_args()
    if args.origin_model:
        import types
        from medusa.model.origin_model import Model,Tokenizer, medusa_generate
        from transformers_stream_generator import init_stream_support
        init_stream_support()
        engine = Model.from_pretrained(args.model)
        tokenizer = Tokenizer.from_pretrained(args.model)
        engine.medusa_generate = types.MethodType(medusa_generate, engine)
        engine.tokenizer = tokenizer
        print("启动原始模型")
    else:
        engine = MedusaModel.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        tokenizer = engine.get_tokenizer()
        print("启动medusa模型")
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)