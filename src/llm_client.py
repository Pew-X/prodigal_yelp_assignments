import os
import json
import time
import re
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class LLMClient:
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant", # can use diffrent models
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        logger.debug(f"Initializing LLMClient with model={model}, max_retries={max_retries}, retry_delay={retry_delay}")
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.error("GROQ_API_KEY environment variable not set")
                raise ValueError("GROQ_API_KEY is not set in environment variables")
            
            self.client = Groq(api_key=api_key)
            self.model = model
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            logger.info(f"LLMClient initialized successfully with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLMClient: {str(e)}")
            raise


    def complete(
        self, system_prompt: str, user_prompt: str,
        temperature: float = 0.0,
    ) -> dict:
        """
        Completes a task using the LLM.
        
        Returns a result dict:
          raw_response  : str | None - Raw response from LLM
          parsed        : dict | None - Parsed JSON response
          json_valid    : bool - Whether JSON parsing succeeded
          latency_ms    : float - Request latency in milliseconds
          error         : str | None - Error message if request failed

          check groq logs dashbaord too
        """
        logger.debug(f"Starting completion request (attempt counter reset, max_retries={self.max_retries})")
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Completion attempt {attempt + 1}/{self.max_retries}")
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
                logger.debug(f"LLM response received in {latency_ms:.2f}ms")
                
                parsed, json_valid = self._parse_json(raw)
                logger.debug(f"JSON parsing result: valid={json_valid}, parsed_keys={list(parsed.keys()) if parsed else None}")

                return {
                    "raw_response": raw,
                    "parsed": parsed,
                    "json_valid": json_valid,
                    "latency_ms": round(latency_ms, 2),
                    "error": None,
                }
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying after {wait_time}s (exponential backoff)")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts exhausted. Final error: {str(e)}")
                    return {
                        "raw_response": None,
                        "parsed": None,
                        "json_valid": False,
                        "latency_ms": 0,
                        "error": str(e),
                    }
                
    def _parse_json(self, text: str) -> tuple:
        """
        json extraction strategy: Direct parse -> Strip markdown code fences -> Regex extract first {...} block
      
        """
        logger.debug(f"Attempting to parse JSON from text (length={len(text)})")
        
        
        try:
            result = json.loads(text)
            logger.debug("[OK] JSON parsed successfully via direct parse")
            return result, True
        except json.JSONDecodeError as e:
            logger.debug(f"[FAIL] Direct parse failed: {str(e)}")

        
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        try:
            result = json.loads(cleaned)
            logger.debug("[OK] JSON parsed successfully after removing markdown fences")
            return result, True
        except json.JSONDecodeError as e:
            logger.debug(f"[FAIL] Fence-stripped parse failed: {str(e)}")

        
        match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                logger.debug("[OK] JSON parsed successfully via regex extraction")
                return result, True
            except json.JSONDecodeError as e:
                logger.debug(f"[FAIL] Regex extraction parse failed: {str(e)}")

        logger.warning("[FAIL] All JSON parsing strategies failed")
        return None, False

                
