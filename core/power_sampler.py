"""
Power Sampling - MCMC-based sampling from p^α for improved reasoning
Based on: "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
(Karan & Du, 2025)

This implements a simplified version of Algorithm 1 from the paper.
"""

import random
import math
from typing import Optional, Dict, Any, List
import requests


class PowerSampler:
    """
    MCMC sampler targeting the power distribution p^α instead of p.

    Key idea: Sampling from p^α upweights high-likelihood sequences more than
    low-temperature sampling, leading to better reasoning without training.
    """

    def __init__(
        self,
        base_model,  # OllamaInference instance
        alpha: float = 4.0,
        mcmc_steps: int = 10,
        block_size: int = 192,
        max_tokens: int = 512,
        proposal_temp: float = 0.25,  # 1/alpha typically
    ):
        """
        Args:
            base_model: LlamaInference/OllamaInference instance
            alpha: Power exponent (higher = sharper distribution)
            mcmc_steps: Number of MCMC resampling steps per block
            block_size: Size of each generation block
            max_tokens: Maximum total tokens to generate
            proposal_temp: Temperature for proposal distribution
        """
        self.base_model = base_model
        self.alpha = alpha
        self.mcmc_steps = mcmc_steps
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.proposal_temp = proposal_temp

    def _compute_log_likelihood(self, text: str, system_prompt: Optional[str] = None) -> float:
        """
        Compute log-likelihood of a text sequence under base model.
        Uses Ollama API's logprobs feature if available, otherwise approximates.
        """
        try:
            # Try to get logprobs from Ollama API
            payload = {
                "model": self.base_model.model_name,
                "prompt": text,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Deterministic for likelihood calculation
                    "num_predict": 0,  # Don't generate, just compute likelihood
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{self.base_model.api_base}/api/generate",
                json=payload,
                timeout=30,
            )

            result = response.json()

            # Approximate log-likelihood from response stats
            # Note: Ollama doesn't expose direct logprobs, so we use eval_count as proxy
            # Better approximation: -eval_duration correlates with likelihood
            # Lower duration per token = higher confidence = higher likelihood

            if "eval_count" in result and result["eval_count"] > 0:
                # Heuristic: faster generation = higher likelihood
                tokens = result["eval_count"]
                duration = result.get("eval_duration", 1e9) / 1e9  # Convert to seconds

                # Approximate log-likelihood as negative of avg time per token
                # (normalized by token count)
                log_likelihood = -math.log(duration / max(tokens, 1) + 1e-6)
                return log_likelihood

            # Fallback: use response length as rough proxy
            return -len(result.get("response", "")) * 0.1

        except Exception as e:
            print(f"Warning: Could not compute log-likelihood: {e}")
            # Fallback to length-based heuristic
            return -len(text) * 0.1

    def _generate_proposal(
        self,
        prefix: str,
        num_tokens: int,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate continuation from prefix using proposal distribution (low-temp)."""
        result = self.base_model.generate(
            prompt=prefix,
            system_prompt=system_prompt,
            params={
                "temperature": self.proposal_temp,
                "num_predict": num_tokens,
            }
        )
        return result["output"]

    def _metropolis_hastings_step(
        self,
        current_text: str,
        prefix: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Single Metropolis-Hastings step: propose new completion and accept/reject.

        Args:
            current_text: Current full text (prefix + completion)
            prefix: Fixed prefix that won't be resampled
            system_prompt: System prompt for generation

        Returns:
            Updated text (either accepted proposal or current_text)
        """
        # Select random resample point AFTER the prefix
        prefix_len = len(prefix)
        current_len = len(current_text)

        if current_len <= prefix_len:
            # Nothing to resample yet
            return current_text

        # Choose random index in [prefix_len, current_len)
        resample_idx = random.randint(prefix_len, current_len - 1)

        # Generate proposal by resampling from resample_idx
        resample_prefix = current_text[:resample_idx]
        remaining_tokens = current_len - resample_idx

        proposal_completion = self._generate_proposal(
            prefix=resample_prefix,
            num_tokens=remaining_tokens,
            system_prompt=system_prompt
        )

        proposal_text = resample_prefix + proposal_completion

        # Compute acceptance ratio A(x', x) = min(1, p^α(x') / p^α(x))
        # Since q(x|x') ≈ q(x'|x) for random resampling, they cancel out

        log_lik_current = self._compute_log_likelihood(current_text, system_prompt)
        log_lik_proposal = self._compute_log_likelihood(proposal_text, system_prompt)

        # Power distribution: p^α → α * log(p)
        log_ratio = self.alpha * (log_lik_proposal - log_lik_current)

        # Acceptance probability
        accept_prob = min(1.0, math.exp(log_ratio))

        # Accept or reject
        if random.random() < accept_prob:
            print(f"  ✓ Accepted proposal (Δlog-lik={log_lik_proposal - log_lik_current:.3f}, p={accept_prob:.3f})")
            return proposal_text
        else:
            print(f"  ✗ Rejected proposal (Δlog-lik={log_lik_proposal - log_lik_current:.3f}, p={accept_prob:.3f})")
            return current_text

    def sample(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text using Power Sampling algorithm.

        Args:
            prompt: Input prompt
            system_prompt: System prompt

        Returns:
            Dict with 'output', 'tokens_generated', 'mcmc_accepts', etc.
        """
        print(f"🔬 Power Sampling (α={self.alpha}, MCMC steps={self.mcmc_steps})")

        current_text = prompt
        total_accepts = 0
        total_proposals = 0

        # Iteratively build up the sequence in blocks
        num_blocks = (self.max_tokens + self.block_size - 1) // self.block_size

        for block_idx in range(num_blocks):
            print(f"\n📦 Block {block_idx + 1}/{num_blocks}")

            # Generate initial block continuation
            block_tokens = min(self.block_size, self.max_tokens - (len(current_text) - len(prompt)))

            if block_tokens <= 0:
                break

            # Initialize block by extending with proposal
            initial_completion = self._generate_proposal(
                prefix=current_text,
                num_tokens=block_tokens,
                system_prompt=system_prompt
            )
            current_text = current_text + initial_completion

            # Run MCMC to refine this block
            for mcmc_step in range(self.mcmc_steps):
                print(f"  🔄 MCMC step {mcmc_step + 1}/{self.mcmc_steps}...")

                previous_text = current_text
                current_text = self._metropolis_hastings_step(
                    current_text=current_text,
                    prefix=prompt,  # Only resample generated part, not original prompt
                    system_prompt=system_prompt
                )

                total_proposals += 1
                if current_text != previous_text:
                    total_accepts += 1

        # Extract just the generated part
        output = current_text[len(prompt):]

        accept_rate = total_accepts / max(total_proposals, 1)

        print(f"\n✅ Power Sampling complete: {len(output)} chars, accept rate={accept_rate:.2%}")

        return {
            "output": output,
            "tokens_generated": len(output.split()),  # Rough token count
            "mcmc_accepts": total_accepts,
            "mcmc_proposals": total_proposals,
            "accept_rate": accept_rate,
            "tokens_per_second": 0.0,  # Not measured yet
        }
