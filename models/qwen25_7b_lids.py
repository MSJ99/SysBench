import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import logging
from typing import Optional

from .lids import register_lids

class qwen25_7b_lids():
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", lids_alpha=0.3):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lids_alpha = lids_alpha
        
        logging.info(f"device_map: {self.model.hf_device_map}")


    def __call__(self, messages):
        formatted_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sys_content = None
        usr_content = None
        for m in messages:
            if m["role"] == "system" and sys_content is None:
                sys_content = m["content"]
            elif m["role"] == "user" and usr_content is None:
                usr_content = m["content"]

        model_inputs = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        model_inputs = {k: v.to(next(self.model.parameters()).device) for k, v in model_inputs.items()}
        offsets = model_inputs["offset_mapping"][0].tolist()
        prompt_len = model_inputs["input_ids"].size(1)

        use_lids = sys_content is not None and usr_content is not None
        sys_start = sys_end = usr_start = usr_end = None

        if use_lids:
            sys_start_c = formatted_chat.find(sys_content)
            usr_start_c = formatted_chat.find(usr_content)

            if sys_start_c == -1 or usr_start_c == -1:
                use_lids = False
            else:
                sys_span = (sys_start_c, sys_start_c + len(sys_content))
                usr_span = (usr_start_c, usr_start_c + len(usr_content))

                def get_token_range(span):
                    c_start, c_end = span
                    idxs = [i for i, (t_start, t_end) in enumerate(offsets) if not (t_end <= c_start or t_start >= c_end)]

                    if not idxs:
                        return None, None
                    return idxs[0], idxs[-1] + 1
                
                sys_start, sys_end = get_token_range(sys_span)
                usr_start, usr_end = get_token_range(usr_span)

                if sys_start is None or usr_start is None:
                    use_lids = False

        generate_inputs = {k: v for k, v in model_inputs.items() if k != "offset_mapping"}
        hooks = []

        if use_lids:
            num_layers = self.model.config.num_hidden_layers
            layer_indices = list(range(num_layers))

            hooks = register_lids(
                self.model,
                layer_indices=layer_indices,
                alpha=self.lids_alpha,
                prompt_len=prompt_len,
                sys_start=sys_start,
                sys_end=sys_end,
                usr_start=usr_start,
                usr_end=usr_end,
            )

        try:
            generated = self.model.generate(
                **generate_inputs,
                max_new_tokens=2048,
                do_sample=False,
            )
        finally:
            for h in hooks:
                h.remove()

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(generate_inputs["input_ids"], generated)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]

    qwen25_7b_lids_model = qwen25_7b_lids()
    print(qwen25_7b_lids_model(messages))

