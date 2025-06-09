import random
from dataclasses import dataclass

import tiktoken
import torch

from learning.gpt2.model import GPT2


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


class NameSampler:
    def __init__(self, names: list[str]):
        self.names = names

    def sample(self, num_names: int) -> list[str]:
        return random.sample(self.names, num_names)


class PromptTemplate:
    def __init__(self, template: str, name_sampler: NameSampler):
        self.template = template
        self.name_sampler = name_sampler

    def sample_abc(self) -> str:
        s1, s2, s3 = self.name_sampler.sample(3)
        return self.template.format(s1=s1, s2=s2, s3=s3)

    def sample_aba(self) -> str:
        s1, s2 = self.name_sampler.sample(2)
        return self.template.format(s1=s1, s2=s2, s3=s1)

    def sample_abb(self) -> str:
        s1, s2 = self.name_sampler.sample(2)
        return self.template.format(s1=s1, s2=s2, s3=s2)

    def from_abc(self, s1, s2, s3) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s3)

    def from_aba(self, s1, s2) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s1)

    def from_abb(self, s1, s2) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s2)


@dataclass
class TopKLogitsResult:
    top_probs: torch.Tensor
    top_indices: torch.Tensor


class IoiCircuitAnalyzer:
    def __init__(
        self,
        model: GPT2,
        tokenizer: tiktoken.Encoding,
        prompt_template: PromptTemplate,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.device = device

    @torch.no_grad()
    def topk_logits(self, prompt: str, k: int) -> TopKLogitsResult:
        self.model.eval()
        indices = [self.tokenizer.encode(prompt)]
        last_output = self.forward(indices)

        # shape: [V] (due to the squeeze)
        V = self.model.config.vocab_size
        probs = torch.softmax(last_output, dim=-1).squeeze(0)
        assert_shape("probs", probs, (V,))

        top_probs, top_indices = torch.topk(probs, k=k)
        assert_shape("top_probs", top_probs, (k,))
        assert_shape("top_indices", top_indices, (k,))

        return TopKLogitsResult(
            top_probs=top_probs,
            top_indices=top_indices,
        )

    @torch.no_grad()
    def forward(self, indices: list[list[int]]):
        self.model.eval()
        B = len(indices)
        S = len(indices[0])
        V = self.model.config.vocab_size

        # shape: [B, S]
        inputs = torch.tensor(indices, dtype=torch.long, device=self.device)
        assert_shape("inputs", inputs, (B, S))

        # shape: [B, S, V]
        outputs = self.model(inputs)
        assert_shape("outputs", outputs, (B, S, V))

        # shape: [B, V]
        last_output = outputs[:, -1, :]
        assert_shape("last_output", last_output, (B, V))

        return last_output

    def capture_baseline_output(self, batch_size: int) -> list[list[torch.Tensor]]:
        """
        Runs the model `batch_size` times with the same template (different names)
        and captures the output of all the heads in all the blocks.

        Returns: the mean of the frozen output. List of blocks of heads of tensors.

        Example:
        ```
        [
            [head_output_1, head_output_2, ...],
            [head_output_1, head_output_2, ...],
        ]
        ```
        """
        self.model.set_capture_output(True)
        self.model.set_use_frozen_output(False)
        cases = [self.prompt_template.sample_abc() for _ in range(batch_size)]
        indices = self.tokenizer.encode_batch(cases)
        self.forward(indices)

        B = batch_size
        S = len(indices[0])
        H = self.model.config.embedding_size // self.model.config.num_heads

        results: list[list[torch.Tensor]] = []
        for block_idx in range(self.model.config.num_blocks):
            block_results: list[torch.Tensor] = []
            for head_idx in range(self.model.config.num_heads):
                output = (
                    self.model.blocks[block_idx].attention.heads[head_idx].frozen_output
                )
                assert_shape("output", output, (B, S, H))

                # shape: [B, S, H] -> [1, S, H]
                mean_output = output.mean(dim=0, keepdim=True)
                assert_shape("mean_output", mean_output, (1, S, H))

                block_results.append(mean_output)

            results.append(block_results)

        self.model.set_capture_output(False)
        self.baseline_output = results
        return results

    def analyze_head(self, block_idx: int, head_idx: int):
        pass
