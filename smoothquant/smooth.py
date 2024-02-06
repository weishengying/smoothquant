import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock

@torch.no_grad()
def smooth_fcs(fcs, act_scales, alpha=0.5):

    from transformers_modules.skywork_13B_chat.modeling_llama import LlamaRMSNorm

    if not isinstance(fcs, list):
        fcs = [fcs]

    for fc in fcs:
        assert isinstance(fc, nn.Linear)

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)
    
    fc.scales = scales
    
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1)) # fc.weight: (*, hidden_states)


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5): # ln: 是layer_norm, fcs: 线形层

    from transformers_modules.skywork_13B_chat.modeling_llama import LlamaRMSNorm

    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, nn.LayerNorm)
    assert isinstance(ln, LlamaRMSNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)
    
    ln.weight.div_(scales) # ln.weight:(hidden_states),  scale: (hidden_states)
    # ln.bias.div_(scales) # LlamaRMSNorm don't have bias
    
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1)) # fc.weight: (*, hidden_states)


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):

    from transformers_modules.skywork_13B_chat.modeling_llama import LlamaMLP, LlamaAttention, LlamaDecoderLayer
    
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            input_layernorm = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj'] #这里只选择 q_proj 的原因是，q_proj, k_proj, v_porj 三个 linear 的输入是相同的，所以这三个linear对应的激活值scale也必定是相同的
            smooth_ln_fcs(input_layernorm, qkv, qkv_input_scales, alpha)
        
            post_layer_norm = module.post_attention_layernorm
            gate_proj = module.mlp.gate_proj
            up_proj = module.mlp.up_proj
            gate_up_projs = [gate_proj, up_proj]
            gate_proj_input_scale = scales[name + '.mlp.gate_proj'] # gate 和 up 两个 linear的输入相同
            smooth_ln_fcs(post_layer_norm, gate_up_projs, gate_proj_input_scale, alpha)

            # 另外两个前面没有 layernorm 的 gemm
            o_proj = module.self_attn.o_proj
            o_proj_input_scale = scales[name + '.self_attn.o_proj']
            smooth_fcs(o_proj, o_proj_input_scale, alpha)

            down_proj = module.mlp.down_proj
            down_proj_input_scale = scales[name + '.mlp.down_proj']
            smooth_fcs(down_proj, down_proj_input_scale, alpha)

        elif isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + '.self_attention.query_key_value']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
