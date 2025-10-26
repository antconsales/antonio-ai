"""
Ollama Wrapper - Alternative to llama.cpp for Antonio
Provides same interface as LlamaInference but uses Ollama API
"""

import requests
import time
from typing import Optional, Dict, Any


class OllamaInference:
    """Wrapper per Ollama API con interfaccia compatibile a LlamaInference"""

    def __init__(
        self,
        model_path: str,  # model name in Ollama format
        api_base: str = "http://localhost:11434",
        default_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_path
        self.api_base = api_base
        
        # Parametri di default ottimizzati per Raspberry Pi 4
        self.default_params = default_params or {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_predict": 128,
        }

        # Verify Ollama is running
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Ollama connection verified")
        except Exception as e:
            raise RuntimeError(f"Ollama not available at {self.api_base}: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Genera una risposta usando Ollama API

        Returns:
            {
                "output": str,
                "tokens_generated": int,
                "tokens_per_second": float,
                "time_elapsed": float,
                "prompt_tokens": int,
            }
        """
        # Merge parametri
        run_params = {**self.default_params, **(params or {})}

        # Build request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": run_params.get("temperature", 0.7),
                "top_p": run_params.get("top_p", 0.9),
                "repeat_penalty": run_params.get("repeat_penalty", 1.05),
                "num_predict": run_params.get("num_predict", 256),
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Execute request
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=120,  # 2 min timeout
            )
            
            elapsed = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()

            # Extract response
            output_text = result.get("response", "").strip()

            # Calculate stats
            prompt_tokens = result.get("prompt_eval_count", 0)
            tokens_generated = result.get("eval_count", 0)
            
            # Calculate tokens/sec
            total_time_ns = result.get("total_duration", 0)
            eval_time_ns = result.get("eval_duration", 0)
            
            tokens_per_second = 0.0
            if eval_time_ns > 0:
                tokens_per_second = (tokens_generated * 1e9) / eval_time_ns

            return {
                "output": output_text,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "time_elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "response_time_ms": elapsed * 1000,
            }

        except requests.exceptions.Timeout:
            raise RuntimeError("Generation timeout (120s)")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def adjust_for_temperature(self, cpu_temp: float):
        """Aggiusta parametri in base a temperatura CPU (Energy-Aware)"""
        if cpu_temp > 75:
            self.default_params["num_predict"] = 80
            print(f"⚠️  CPU temp {cpu_temp}°C - reduced max tokens to 128")
        elif cpu_temp > 70:
            self.default_params["num_predict"] = 100
            print(f"⚠️  CPU temp {cpu_temp}°C - reduced max tokens to 192")
        else:
            self.default_params["num_predict"] = 256


# For backward compatibility, alias as LlamaInference
LlamaInference = OllamaInference
