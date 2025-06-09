import random
from dataclasses import dataclass

import tiktoken
import torch

from learning.gpt2.metrics import IoiMetrics
from learning.gpt2.model import GPT2


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@dataclass
class HeadAnalysisResult:
    """Results from analyzing a specific attention head"""

    original_probs: torch.Tensor
    patched_probs: torch.Tensor
    metrics: IoiMetrics


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
class TopKProbsResult:
    top_probs: torch.Tensor
    top_indices: torch.Tensor


@dataclass
class CapturedOutput:
    # Mean of the head outputs
    # indexed by block_idx and head_idx
    head_outputs: list[list[torch.Tensor]]

    # Mean of the final probs
    probs: torch.Tensor


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

        self.baseline_output: CapturedOutput | None = None

        self.V = self.model.config.vocab_size
        self.H = self.model.config.embedding_size // self.model.config.num_heads

    def topk_probs(self, prompt: str, k: int) -> TopKProbsResult:
        indices = [self.tokenizer.encode(prompt)]
        probs = self.forward(indices)
        assert_shape("probs", probs, (1, self.V))

        top_probs, top_indices = torch.topk(probs.squeeze(0), k=k)
        assert_shape("top_probs", top_probs, (k,))
        assert_shape("top_indices", top_indices, (k,))

        return TopKProbsResult(
            top_probs=top_probs,
            top_indices=top_indices,
        )

    @torch.no_grad()
    def forward(self, indices: list[list[int]]):
        self.model.eval()
        B = len(indices)
        S = len(indices[0])

        # shape: [B, S]
        inputs = torch.tensor(indices, dtype=torch.long, device=self.device)
        assert_shape("inputs", inputs, (B, S))

        # shape: [B, S, V]
        outputs = self.model(inputs)
        assert_shape("outputs", outputs, (B, S, self.V))

        # shape: [B, V]
        last_output = outputs[:, -1, :]
        assert_shape("last_output", last_output, (B, self.V))

        # shape: [B, V]
        probs = torch.softmax(last_output, dim=-1)
        assert_shape("probs", probs, (B, self.V))

        return probs

    def capture_baseline_output(self, batch_size: int):
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

        B = batch_size
        S = len(indices[0])

        # shape: [V]
        probs = self.forward(indices)
        assert_shape("probs", probs, (B, self.V))

        mean_probs = probs.mean(dim=0, keepdim=True)
        assert_shape("mean_probs", mean_probs, (1, self.V))

        results: list[list[torch.Tensor]] = []
        for block_idx in range(self.model.config.num_blocks):
            block_results: list[torch.Tensor] = []
            for head_idx in range(self.model.config.num_heads):
                output = (
                    self.model.blocks[block_idx].attention.heads[head_idx].frozen_output
                )
                assert_shape("output", output, (B, S, self.H))

                # shape: [B, S, H] -> [1, S, H]
                mean_output = output.mean(dim=0, keepdim=True)
                assert_shape("mean_output", mean_output, (1, S, self.H))

                block_results.append(mean_output)

            results.append(block_results)

        self.model.set_capture_output(False)
        self.baseline_output = CapturedOutput(
            head_outputs=results,
            probs=mean_probs,
        )

    def analyze_head(
        self, block_idx: int, head_idx: int, s1: str, s2: str, s3: str
    ) -> HeadAnalysisResult:
        if self.baseline_output is None:
            raise ValueError("Must call `capture_baseline_output` first")

        # Phase B: capture the output before path patching
        self.model.set_capture_output(True)
        self.model.set_use_frozen_output(False)
        indices = [self.tokenizer.encode(self.prompt_template.from_abc(s1, s2, s3))]

        S = len(indices[0])

        original_probs = self.forward(indices)
        assert_shape("original_probs", original_probs, (1, self.V))

        # Phase C: path patching head[block_idx][head_idx] using the baseline output
        self.model.set_capture_output(False)
        self.model.set_use_frozen_output(True)
        captured_output = self.baseline_output.head_outputs[block_idx][head_idx]
        assert_shape("captured_output", captured_output, (1, S, self.H))

        self.model.blocks[block_idx].attention.heads[
            head_idx
        ].frozen_output = captured_output

        patched_probs = self.forward(indices)
        assert_shape("patched_probs", patched_probs, (1, self.V))

        # Analysis is done, reset `use_frozen_output` to False
        self.model.set_use_frozen_output(False)

        # Flatten for metric computation
        original_probs = original_probs.squeeze(0)
        patched_probs = patched_probs.squeeze(0)

        s1_idx = indices[0][0]
        s2_idx = indices[0][1]
        ioi_metrics = IoiMetrics.from_probs(
            original_probs, patched_probs, s1_idx, s2_idx
        )

        return HeadAnalysisResult(
            original_probs=original_probs,
            patched_probs=patched_probs,
            metrics=ioi_metrics,
        )
