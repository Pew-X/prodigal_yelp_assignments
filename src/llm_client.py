import os
import json
import time
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()



class LLMClient:
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant", # can use diffrent models, e.g. "groq-2b", "llama-3.1-8b-instant", "groq-8b"
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay


    def complete(
        self, system_prompt: str, user_prompt: str,
        temperature: float = 0.0,
    ) -> dict:
        """
        Returns a result dict:
          raw_response  : str | None
          parsed        : dict | None
          json_valid    : bool
          latency_ms    : float
          error         : str | None
        """
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=300,
                )
                latency_ms = (time.time() - start) * 1000
                raw = response.choices[0].message.content.strip()
                parsed, json_valid = self._parse_json(raw) #parsing json here 

                return {
                    "raw_response": raw,
                    "parsed": parsed,
                    "json_valid": json_valid,
                    "latency_ms": round(latency_ms, 2),
                    "error": None,
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))  # exponential backoff
                else:
                    return {
                        "raw_response": None,
                        "parsed": None,
                        "json_valid": False,
                        "latency_ms": 0,
                        "error": str(e),
                    }
                
    def _parse_json(self, text: str) -> tuple:
        """
        json extraction strategy Direct parse/Strip markdown code fences, retry / 
        Regex extract first {...} block
        """
        #direct
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            pass

        #strip fences
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        try:
            return json.loads(cleaned), True
        except json.JSONDecodeError:
            pass

        #extract first JSON object
        match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()), True
            except json.JSONDecodeError:
                pass

        return None, False

                
