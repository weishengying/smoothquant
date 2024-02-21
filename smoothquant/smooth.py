import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock

@torch.no_grad()
def smooth_fc_fc(fc_1, fc_2, act_max_values, alpha=0.5):

    assert isinstance(fc_1, nn.Linear)
    assert isinstance(fc_2, nn.Linear)
    assert fc_1.out_features == fc_2.in_features == act_max_values.numel()

    if not isinstance(fc_2, list):
        fcs = [fc_2]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_max_values = act_max_values.to(device=device, dtype=dtype)
    weight_max_values = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_max_values = weight_max_values.max(dim=0)[0].clamp(min=1e-5)

    smmoth_scales = (act_max_values.pow(alpha) / weight_max_values.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)
    
    fc_1.weight.div_(smmoth_scales.view(-1, 1))
    fc_2.weight.mul_(smmoth_scales)

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_max_values, alpha=0.5): # ln: 是layer_norm, fcs: 线形层, act_max_values: 激活值每个 channel 的最大值

    from transformers_modules.moe.modeling_skywork_moe import MixtralRMSNorm

    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, MixtralRMSNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_max_values.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_max_values = act_max_values.to(device=device, dtype=dtype)
    weight_max_values = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_max_values = weight_max_values.max(dim=0)[0].clamp(min=1e-5)

    smmoth_scales = (act_max_values.pow(alpha) / weight_max_values.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)
    
    ln.weight.div_(smmoth_scales) # ln.weight:(hidden_states),  smmoth_scales: (hidden_states)
    
    for fc in fcs:
        fc.weight.mul_(smmoth_scales.view(1, -1)) # fc.weight: (*, hidden_states)

@torch.no_grad()
def smooth_ln_gate_fcs(ln, gate, fcs, act_max_values, alpha=0.5):

    from transformers_modules.moe.modeling_skywork_moe import MixtralRMSNorm

    if not isinstance(fcs, list):
        fcs = [fcs]

    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert isinstance(ln, MixtralRMSNorm)
        assert ln.weight.numel() == fc.in_features == act_max_values.numel()
        assert isinstance(gate, nn.Linear)
        assert gate.in_features== fc.in_features == act_max_values.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_max_values = act_max_values.to(device=device, dtype=dtype)
    weight_max_values = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_max_values = weight_max_values.max(dim=0)[0].clamp(min=1e-5)

    smmoth_scales = (act_max_values.pow(alpha) / weight_max_values.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)
    
    gate.weight.mul_(smmoth_scales.view(1, -1))
    ln.weight.div_(smmoth_scales) # ln.weight:(hidden_states),  scale: (hidden_states)
    
    for fc in fcs:
        fc.weight.mul_(smmoth_scales.view(1, -1)) # fc.weight: (*, hidden_states)


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):

    from transformers_modules.moe.modeling_skywork_moe import MixtralDecoderLayer, MixtralSdpaAttention, MixtralSparseMoeBlock, MixtralBLockSparseTop2MLP

    for name, module in model.named_modules():
        if isinstance(module, MixtralDecoderLayer):
            # smooth attention 中的 qkv proj
            input_layernorm = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj'] #q_proj, k_proj, v_porj 三个 linear 的输入是相同的
            smooth_ln_fcs(input_layernorm, qkv, qkv_input_scales, alpha) # smooth qkv proj

            # smooth mlp 中的 w1 和 w3 proj
            gate = module.block_sparse_moe.gate
            post_layer_norm = module.post_attention_layernorm
            w1_w3_projs = []
            for i in range(len(module.block_sparse_moe.experts)):
                expert = module.block_sparse_moe.experts[i]
                assert isinstance(expert, MixtralBLockSparseTop2MLP)
                w1_proj = expert.w1
                w3_proj = expert.w3
                w1_w3_projs.append(w1_proj)
                w1_w3_projs.append(w3_proj)
            w1_proj_input_scale = scales[name + f'.block_sparse_moe.experts.0.w1'] # 每个专家 w1 和 w3 两个 linear的输入"相同"
            smooth_ln_gate_fcs(post_layer_norm, gate, w1_w3_projs, w1_proj_input_scale, alpha) # smooth mlp 中的 w1 和 w3 proj

            # smooth mlp 中的 w2_proj
            for i in range(len(module.block_sparse_moe.experts)):
                expert = module.block_sparse_moe.experts[i]
                assert isinstance(expert, MixtralBLockSparseTop2MLP)
                w2_proj = expert.w2
                w3_proj = expert.w3
                w2_proj_input_scale = scales[name + f'.block_sparse_moe.experts.{i}.w2'] 
                smooth_fc_fc(w3_proj, w2_proj, w2_proj_input_scale, alpha) # smooth mlp 中的 w2