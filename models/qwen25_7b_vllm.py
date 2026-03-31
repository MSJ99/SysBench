from openai import OpenAI


class qwen25_7b_vllm:
    def __init__(self, api_base="http://localhost:33619/v1", api_key="custom-key"):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.kwargs = {"temperature": 0, "seed": 42, "max_tokens": 2048}

    def __call__(self, messages):
        for _ in range(100):
            try:
                r = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.kwargs,
                )
                return r.choices[0].message.content
            except Exception as e:
                print(e)
                continue
        return None


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    model = qwen25_7b_vllm()
    print(model(messages))
