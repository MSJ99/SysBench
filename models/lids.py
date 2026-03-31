import logging
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig


def register_lids(
    model,
    layer_indices,
    alpha,
    prompt_len,
    sys_start,
    sys_end,
    usr_start,
    usr_end,
):
    hooks = []

    def make_hook():
        def lids_gen(module, module_input, module_output):
            # module_output: Tensor or (Tensor, ...)
            hidden_states = (
                module_output[0] if isinstance(module_output, tuple) else module_output
            )
            # hidden_states: [batch, seq_len, hidden_dim]
            seq_len = hidden_states.shape[1]
            if seq_len == prompt_len:
                hs = hidden_states.clone()
                ct_sys = hs[:, sys_start:sys_end, :].mean(dim=1)  # [batch, hidden]
                ct_usr = hs[:, usr_start:usr_end, :].mean(dim=1)  # [batch, hidden]

                direction = ct_sys - ct_usr  # [batch, hidden]

                # suppose batch_size=1
                direction = direction[0]  # [hidden]

                d = direction.view(1, 1, -1)  # [1,1,hidden] → broadcast

                hs[:, sys_start:sys_end, :] += alpha * d
                hs[:, usr_start:usr_end, :] -= alpha * d

                if isinstance(module_output, tuple):
                    return (hs,) + module_output[1:]
                else:
                    return hs

            return module_output

        return lids_gen

    for layer_idx in layer_indices:
        hook_fn = make_hook()
        h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        hooks.append(h)

    return hooks