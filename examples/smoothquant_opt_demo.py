import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
import sys

os.environ["CUDA_VISIBLDE_DEVICES"]='0'

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=False):
    
    from transformers_modules.skywork_13B_chat.modeling_llama import LlamaMLP, LlamaAttention

    for name, m in model.model.named_modules():
        
        if isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.up_proj = W8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant)
            if hasattr(m.down_proj, "scales"):
                m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant, scales=m.down_proj.scales)
            else:
                m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            if hasattr(m.o_proj, "scales"):
                m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant, scales=m.o_proj.scales)
            else:
                m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0) #(1, seq_len)
            label = input_ids[:, -1] # 最后一个 token 当作 label
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :] # 倒数第二个 token
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

from datasets import load_dataset

model_path = "/mnt/infra/weishengying/model/skywork_13B_chat"
sys.path.append(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
data_files = {"validation": "validation-00000-of-00001.parquet"}
dataset = load_dataset('../dataset/lambada', data_files=data_files, split='validation[1000:2000]')
evaluator = Evaluator(dataset, tokenizer, 'cuda')

model_fp16 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
acc_fp16 = evaluator.evaluate(model_fp16)
print(f'Original model (fp16) accuracy: {acc_fp16}')


model_w8a8 = quantize_model(model_fp16)
print(model_w8a8)
acc_w8a8 = evaluator.evaluate(model_w8a8)
print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')


model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
act_scales = torch.load('act_scales/chat-13b-feilun.pt')
smooth_lm(model, act_scales, 0.5)
model_smoothquant_w8a8 = quantize_model(model)
print(model_smoothquant_w8a8)

acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')
