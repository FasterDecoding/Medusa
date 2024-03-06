import torch
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from .medusa_model import MedusaConfig


class Tokenizer():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
        model_dir=config.base_model_name_or_path
        return AutoTokenizer.from_pretrained(model_dir,
                                            use_fast=True,
                                            trust_remote_code=True)


class Model():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
        model_dir=config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                    device_map="auto",
                                                    torch_dtype=torch.float16,
                                                    trust_remote_code=True)

        model.generation_config = GenerationConfig.from_pretrained(model_dir)
        return model
    
def medusa_generate(self, **kwargs):
    output_ids = None
    kwargs['max_length'] = kwargs['max_steps']+kwargs['input_ids'].shape[-1]
    generator = self.generate(**kwargs, do_stream=True, do_sample=True)
    for tokens in generator:
        tokens=tokens.unsqueeze(-1)
        if output_ids is None:
            output_ids = tokens
        else:
            output_ids = torch.cat((output_ids, tokens), dim=-1)
        decoded_texts = self.tokenizer.batch_decode(output_ids, 
                                               skip_special_tokens=True,
                                               spaces_between_special_tokens=False,
                                               clean_up_tokenization_spaces=True,)
        yield {"text": decoded_texts}