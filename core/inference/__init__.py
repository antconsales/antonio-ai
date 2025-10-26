"""
Inference Engine - Wrapper per Ollama API + confidence scoring
"""

from .ollama_wrapper import LlamaInference
from .confidence import ConfidenceScorer

__all__ = [
    "LlamaInference",
    "ConfidenceScorer",
]
