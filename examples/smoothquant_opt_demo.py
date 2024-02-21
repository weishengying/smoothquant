import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A16Linear, W4A16Linear, W8A8Linear
import sys
import argparse
from tqdm import tqdm
from smoothquant.calibration import get_act_scales
from pathlib import Path

os.environ["CUDA_VISIBLDE_DEVICES"]='0, 1'

def quantize_model(model, A_W_max_values, weight_quant='per_channel', act_quant='per_tensor', quantize_bmm_input=False):
    
    from transformers_modules.moe.modeling_skywork_moe import MixtralSdpaAttention, MixtralSparseMoeBlock, MixtralBLockSparseTop2MLP

    for name, m in tqdm(model.model.named_modules(), desc="Quanting", unit="layer", total=model.config.num_hidden_layers):
            if isinstance(m, MixtralSparseMoeBlock):
                for i in range(len(m.experts)):
                    expert = m.experts[i]
                    assert isinstance(expert, MixtralBLockSparseTop2MLP)
                    expert.w1 = W4A16Linear.from_float(expert.w1, weight_quant=weight_quant)
                    expert.w2 = W4A16Linear.from_float(expert.w2, weight_quant=weight_quant)
                    expert.w3 = W4A16Linear.from_float(expert.w3, weight_quant=weight_quant)
            elif isinstance(m, MixtralSdpaAttention):
                pass
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                qkv_input_max_values = A_W_max_values[f"model.{name}.q_proj"]
                o_input_max_values = A_W_max_values[f"model.{name}.o_proj"]
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj, weight_quant="per_tensor", act_quant="per_tensor", quantize_output=quantize_bmm_input)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant="per_tensor", act_quant="per_tensor", quantize_output=quantize_bmm_input)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant="per_tensor", act_quant="per_tensor", quantize_output=quantize_bmm_input)
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
        # for batch in self.dataset:
        
        num_samples = len(self.dataset)
        print(f"num_samples: {num_samples}")
        for i in tqdm(range(num_samples), desc="evaluate, Processing"):
            input_ids = self.dataset[i]['input_ids'].to(self.device).unsqueeze(0) #(1, seq_len)
            label = input_ids[:, -1] # 最后一个 token 当作 label
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :] # 倒数第二个 token
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--save_dir', type=str, help='save smooth model')
    args = parser.parse_args()
    return args

args = parse_args()
if args.save_dir:
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

model_path = args.model_path
sys.path.append(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
data_files = {"validation": "validation-00000-of-00001.parquet"}
dataset = load_dataset('../dataset/lambada', data_files=data_files, split='validation[1000:2000]')
evaluator = Evaluator(dataset, tokenizer, 'cuda')

# model_fp16 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
# acc_fp16 = evaluator.evaluate(model_fp16)
# print(f'Original model (fp16) accuracy: {acc_fp16}')

# A_W_max_values = torch.load('act_scales/8x7B_MoE.pt') # activate and wight max values
# model_w8a8 = quantize_model(model_fp16, A_W_max_values, smooth_quant=False)
# print(model_w8a8)
# acc_w8a8 = evaluator.evaluate(model_w8a8)
# print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
A_W_max_values = torch.load('act_scales/8x7B_MoE.pt')
smooth_lm(model, A_W_max_values, 0.5)
model.save_pretrained(args.save_dir)

# model_smoothquant = quantize_model(model, A_W_max_values)
# print(model_smoothquant)
# acc_smoothquant = evaluator.evaluate(model_smoothquant)
# print(f'SmoothQuant W4A16 quantized model accuracy: {acc_smoothquant}')




'''
python smoothquant_opt_demo.py --model-path /mnt/infra/weishengying/model/moe --save_dir ./skywork_mixtral_smooth
'''