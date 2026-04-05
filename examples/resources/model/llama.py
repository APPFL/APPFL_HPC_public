import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def get_model(model_name: str):
    """Load the model with proper initialization"""
    def skip(*args, **kwargs):
        pass
    
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token='' # provide your Hugging Face token here if the model is private
    )

    # Set sequence length based on model config
    if hasattr(model.config, 'max_position_embeddings'):
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 2048
    
    model.config.use_cache = False
    return model