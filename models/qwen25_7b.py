import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import logging

class qwen25_7b():
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logging.info(f"device_map: {self.model.hf_device_map}")


    def __call__(self, messages):
        formatted_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        model_inputs = {key: tensor.to(next(self.model.parameters()).device) for key, tensor in model_inputs.items()}

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]

    qwen25_7b_model = qwen25_7b()
    print(qwen25_7b_model(messages))

