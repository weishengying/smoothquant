import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from smoothquant.calibration import get_act_scales

def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential", "trust_remote_code": args.trust_remote_code}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-13b', help='model name')
    parser.add_argument('--output-path', type=str, default='act_scales/opt-13b.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-pah', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--trust-remote-code', type=bool, default=False)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args)
    
    data_files = {"validation": "validation-00000-of-00001.parquet"}
    dataset = load_dataset('../dataset/lambada', data_files=data_files, split='validation[:1000]')

    act_scales = get_act_scales(model, tokenizer, dataset,
                                args.num_samples, args.seq_len)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()

'''
python generate_act_scales.py --model-name /mnt/lichang.zhang/skywork-13b-fl-sft-8k-1128-hg/iter_0001200 --output-path act_scales/chat-13b-feilun.pt --trust-remote-code True
'''