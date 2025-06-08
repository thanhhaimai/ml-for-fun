from dataclasses import dataclass

import tiktoken
import torch

from learning.gpt2.model import GPT2


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@dataclass
class TopKLogitsResult:
    top_probs: torch.Tensor
    top_indices: torch.Tensor


@dataclass
class HeadId:
    block_idx: int
    head_idx: int


class IoiCircuitAnalyzer:
    def __init__(self, model: GPT2, tokenizer: tiktoken.Encoding, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def topk_logits(self, prompt: str, k: int) -> TopKLogitsResult:
        self.model.eval()
        indices = self.tokenizer.encode(prompt)
        S = len(indices)

        # shape: [B, S]
        inputs = torch.tensor([indices], dtype=torch.long, device=self.device)
        assert_shape("inputs", inputs, (1, S))

        # shape: [B, S, V]
        outputs = self.model(inputs)
        assert_shape("outputs", outputs, (1, S, self.model.config.vocab_size))

        # shape: [B, V]
        last_output = outputs[:, -1, :]
        assert_shape("last_output", last_output, (1, self.model.config.vocab_size))

        # shape: [V] (due to the squeeze)
        probs = torch.softmax(last_output, dim=-1).squeeze(0)
        assert_shape("probs", probs, (self.model.config.vocab_size,))

        top_probs, top_indices = torch.topk(probs, k=k)
        assert_shape("top_probs", top_probs, (k,))
        assert_shape("top_indices", top_indices, (k,))

        return TopKLogitsResult(
            top_probs=top_probs,
            top_indices=top_indices,
        )
