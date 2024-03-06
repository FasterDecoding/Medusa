import torch
import pdb
import types
from medusa.model.origin_model import Model,Tokenizer, medusa_generate
from transformers_stream_generator import init_stream_support
init_stream_support()


prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {0} ASSISTANT:"
prompt = ["openai是家什么公司？", "2+2等于几？"]
prompt = [prefix.format(p) for p in prompt]
model_dir='/mnt/wx/.cache/huggingface/hub/models--FasterDecoding--medusa-vicuna-7b-v1.3/snapshots/82ac200bf7502419cb49a9e0adcbebe3d1d293f1/'
model = Model.from_pretrained(model_dir)
tokenizer = Tokenizer.from_pretrained(model_dir)
model_inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
# 给实例对象添加方法
model.tokenizer = tokenizer
model.medusa_generate = types.MethodType(medusa_generate, model)
input_ids = model_inputs['input_ids'].to(model.device)
attention_mask = model_inputs['attention_mask'].to(model.device)
generator = model.medusa_generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                temperature=0.1,
                                max_steps=20,
                                top_p=0.8)
for token in generator:  
    print(token['text'])
    

