import os
import re
import time
import threading
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class gpt4o(object):
    _lock = threading.Lock()
    _call_times = []
    RPM_LIMIT = 3

    def __init__(self, model_name="gpt-4o-2024-05-13", key=None) -> None:
        api_key = key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to .env or pass key=...")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"model_name: {self.model_name}")

    def _wait_for_rate_limit(self):
        with gpt4o._lock:
            now = time.time()
            gpt4o._call_times = [t for t in gpt4o._call_times if now - t < 60]
            if len(gpt4o._call_times) >= gpt4o.RPM_LIMIT:
                sleep_time = 60 - (now - gpt4o._call_times[0]) + 0.5
                if sleep_time > 0:
                    time.sleep(sleep_time)
            gpt4o._call_times.append(time.time())

    @staticmethod
    def _parse_retry_after(error_msg):
        msg = str(error_msg)
        total = 0.0
        m = re.search(r'Please try again in (?:(\d+)m)?(\d+(?:\.\d+)?)(m?s)', msg)
        if not m:
            return 21.0
        if m.group(1):
            total += int(m.group(1)) * 60
        value = float(m.group(2))
        unit = m.group(3)
        if unit == 'ms':
            total += value / 1000
        elif unit == 's':
            total += value
        return total + 1.0

    def __call__(self, query, retry=10, temperature=None):
        if isinstance(query, str):
            messages = [{"role":"user","content": query}]
        elif isinstance(query, list):
            messages = query
        else:
            raise ValueError("query must be str or list")
        i = 0
        while i < retry:
            self._wait_for_rate_limit()
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    seed=42,
                )

                result = response.choices[0].message.content

                assert isinstance(result, str) and response.choices[0].finish_reason == "stop"
                return result
            except Exception as e:
                print(e)
                if "429" in str(e):
                    time.sleep(self._parse_retry_after(e))
                i += 1
        raise RuntimeError(f"Failed after {retry} retries")

if __name__ == "__main__":
    gpt4 = gpt4o()
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]
    print(gpt4(messages))
    
    
