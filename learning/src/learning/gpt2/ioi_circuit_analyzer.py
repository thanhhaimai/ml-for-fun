import tiktoken
import torch

from learning.gpt2.model import GPT2


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


class IoiCircuitAnalyzer:
    def __init__(self, model: GPT2, tokenizer: tiktoken.Encoding, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def topk_logits(self, prompt: str, k: int):
        self.model.eval()
        with torch.no_grad():
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
            for i in range(k):
                token = self.tokenizer.decode([int(top_indices[i])])
                print(f"{top_probs[i]:.2f} {token}")

    def analyze(self):
        pass
