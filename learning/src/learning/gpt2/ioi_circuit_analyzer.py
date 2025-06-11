import random
from dataclasses import dataclass
from typing import NamedTuple

import tiktoken
import torch

from learning.gpt2.metrics import DiffLogitsMetrics, ProbsMetrics
from learning.gpt2.model import GPT2


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


class NameSample(NamedTuple):
    names: list[str]
    indices: list[int]


class NameSampler:
    def __init__(self, names: list[str], tokenizer: tiktoken.Encoding):
        self.names = names
        self.indices = [tokenizer.encode(f" {name.strip()}")[0] for name in names]

    def sample(self, num_names: int) -> list[str]:
        return random.sample(self.names, num_names)

    def sample_batch(self, num_names: int, batch_size: int) -> list[NameSample]:
        batches = []
        for _ in range(batch_size):
            sample_indices = random.sample(range(len(self.indices)), num_names)
            names = [self.names[i] for i in sample_indices]
            indices = [self.indices[i] for i in sample_indices]

            batches.append(NameSample(names, indices))

        return batches


class PromptBatch(NamedTuple):
    # The prompts uses s1, s2, s3 as placeholders
    # The indices are the indices of the s1, s2, s3 tokens
    prompts: list[str]
    # shape: [B]
    s1_indices: torch.Tensor
    # shape: [B]
    s2_indices: torch.Tensor
    # shape: [B]
    s3_indices: torch.Tensor


class PromptTemplate:
    def __init__(self, template: str, name_sampler: NameSampler, device: torch.device):
        self.template = template
        self.name_sampler = name_sampler
        self.device = device

    def sample_batch_abc(self, batch_size: int) -> PromptBatch:
        name_samples = self.name_sampler.sample_batch(3, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        s3_indices = []
        for name_sample in name_samples:
            s1, s2, s3 = name_sample.names
            s1_idx = name_sample.indices[0]
            s2_idx = name_sample.indices[1]
            s3_idx = name_sample.indices[2]
            prompts.append(self.template.format(s1=s1, s2=s2, s3=s3))
            s1_indices.append(s1_idx)
            s2_indices.append(s2_idx)
            s3_indices.append(s3_idx)

        return PromptBatch(
            prompts,
            torch.tensor(s1_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
            torch.tensor(s3_indices, device=self.device),
        )

    def sample_batch_aba(self, batch_size: int) -> PromptBatch:
        name_samples = self.name_sampler.sample_batch(2, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        for name_sample in name_samples:
            s1, s2 = name_sample.names
            s1_idx = name_sample.indices[0]
            s2_idx = name_sample.indices[1]
            prompts.append(self.template.format(s1=s1, s2=s2, s3=s1))
            s1_indices.append(s1_idx)
            s2_indices.append(s2_idx)

        return PromptBatch(
            prompts,
            torch.tensor(s1_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
            torch.tensor(s1_indices, device=self.device),
        )

    def sample_batch_abb(self, batch_size: int) -> PromptBatch:
        name_samples = self.name_sampler.sample_batch(2, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        for name_sample in name_samples:
            s1, s2 = name_sample.names
            s1_idx = name_sample.indices[0]
            s2_idx = name_sample.indices[1]
            prompts.append(self.template.format(s1=s1, s2=s2, s3=s2))
            s1_indices.append(s1_idx)
            s2_indices.append(s2_idx)

        return PromptBatch(
            prompts,
            torch.tensor(s1_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
        )

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
class CapturedOutputV1:
    # Mean of the head outputs
    # indexed by block_idx and head_idx
    head_outputs: list[list[torch.Tensor]]

    # Mean of the final probs
    probs: torch.Tensor


@dataclass
class CapturedOutput:
    # Mean of the head outputs
    # indexed by block_idx and head_idx
    head_outputs: list[list[torch.Tensor]]

    # Mean of the final logits
    logits: torch.Tensor


class HeadId(NamedTuple):
    block_idx: int
    head_idx: int


@dataclass
class PathPatchingConfig:
    batch_size: int
    start_head: HeadId
    end_heads: list[HeadId]


@dataclass
class HeadAnalysisResult:
    original_probs: torch.Tensor
    patched_probs: torch.Tensor
    s1_indices: torch.Tensor
    s2_indices: torch.Tensor
    # metrics: DiffLogitsMetrics


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

        self.V = self.model.config.vocab_size
        self.H = self.model.config.embedding_size // self.model.config.num_heads

    def topk_probs(self, prompt: str, k: int) -> TopKProbsResult:
        logits = self.forward([prompt])
        assert_shape("logits", logits, (1, self.V))

        probs = torch.softmax(logits, dim=-1)
        assert_shape("probs", probs, (1, self.V))

        top_probs, top_indices = torch.topk(probs.squeeze(0), k=k)
        assert_shape("top_probs", top_probs, (k,))
        assert_shape("top_indices", top_indices, (k,))

        return TopKProbsResult(
            top_probs=top_probs,
            top_indices=top_indices,
        )

    @torch.no_grad()
    def forward(self, prompts: list[str]):
        self.model.eval()
        indices = self.tokenizer.encode_batch(prompts)
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

        return last_output

    def capture_baseline_output(self, prompts_abc: list[str]) -> CapturedOutput:
        B = len(prompts_abc)
        self.model.set_capture_output(True)
        self.model.set_use_frozen_output(False)

        logits_abc = self.forward(prompts_abc)
        assert_shape("logits_abc", logits_abc, (B, self.V))

        mean_logits_abc = logits_abc.mean(dim=0, keepdim=True)
        assert_shape("mean_logits_abc", mean_logits_abc, (1, self.V))

        results = []
        for block_idx in range(self.model.config.num_blocks):
            block_results = []
            for head_idx in range(self.model.config.num_heads):
                head_output = (
                    self.model.blocks[block_idx].attention.heads[head_idx].frozen_output
                )
                S = head_output.shape[1]
                assert_shape("head_output", head_output, (B, S, self.H))

                mean_output = head_output.mean(dim=0, keepdim=True)
                assert_shape("mean_output", mean_output, (1, S, self.H))

                block_results.append(mean_output)
            results.append(block_results)

        return CapturedOutput(
            head_outputs=results,
            logits=mean_logits_abc,
        )

    def path_patching(
        self,
        config: PathPatchingConfig,
        baseline_output: CapturedOutput,
        prompts_abb: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ================================
        # Phase A: capture the ABC output over all prompts in prompts_abc
        # ================================
        # We're only interested in the output of the start head
        baseline = baseline_output.head_outputs[config.start_head.block_idx][
            config.start_head.head_idx
        ]
        B = len(prompts_abb)
        S = baseline.shape[1]
        assert_shape("baseline", baseline, (1, S, self.H))

        # ================================
        # Phase B: capture the ABB output
        # ================================
        B = len(prompts_abb)
        self.model.set_capture_output(True)
        self.model.set_use_frozen_output(False)
        logits_prepatched = self.forward(prompts_abb)
        assert_shape("logits_prepatched", logits_prepatched, (B, self.V))

        # ================================
        # Phase C: patch the start_head using the ABC start_head output
        # ================================
        # Effectively, we're knocking out the start_head output
        # If the head is not important, the patched output should be close to the ABB output
        # NOTE: the patched output is not used within this phase (it is used in Phase D)
        baseline = baseline.expand(B, S, self.H)
        assert_shape("baseline", baseline, (B, S, self.H))

        self.model.blocks[config.start_head.block_idx].attention.heads[
            config.start_head.head_idx
        ].frozen_output = baseline

        # Run the model using the frozen outputs
        # But also capture all end_heads outputs
        # NOTE: the captured end_heads outputs are not used within this phase (they are used in Phase D)
        self.model.set_capture_output(False)
        for end_head in config.end_heads:
            self.model.blocks[end_head.block_idx].attention.heads[
                end_head.head_idx
            ].should_capture_output = True
        self.model.set_use_frozen_output(True)
        logits_patched = self.forward(prompts_abb)
        assert_shape("logits_patched", logits_patched, (B, self.V))

        # ================================
        # Phase D: Forward with the end_heads frozen output
        # ================================
        if config.end_heads:
            self.model.set_capture_output(False)
            self.model.set_use_frozen_output(True)
            logits_patched = self.forward(prompts_abb)
            assert_shape("patched_probs", logits_patched, (B, self.V))

        self.model.set_capture_output(False)
        self.model.set_use_frozen_output(False)

        return logits_prepatched, logits_patched

    def analyze_head(
        self, config: PathPatchingConfig, baseline_output: CapturedOutput
    ) -> ProbsMetrics:
        B = config.batch_size

        batch = self.prompt_template.sample_batch_abb(config.batch_size)

        logits_prepatched, logits_patched = self.path_patching(
            config,
            baseline_output,
            batch.prompts,
        )
        assert_shape("logits_prepatched", logits_prepatched, (B, self.V))
        assert_shape("logits_patched", logits_patched, (B, self.V))

        return ProbsMetrics.from_logits(
            original_logits=logits_prepatched,
            patched_logits=logits_patched,
            s1_indices=batch.s1_indices,
            s2_indices=batch.s2_indices,
        )

    def analyze_head_pairwise(self, config: PathPatchingConfig) -> DiffLogitsMetrics:
        s1_logits: list[float] = []
        s2_logits: list[float] = []
        s1_probs: list[float] = []
        s2_probs: list[float] = []

        for s1, s2, s3 in self.prompt_template.name_sampler.sample_batch(
            3, config.batch_size
        ):
            s1 = "Mary"
            s2 = "John"
            s3 = "Jerry"

            s1_idx = self.tokenizer.encode(f" {s1.strip()}")[0]
            s2_idx = self.tokenizer.encode(f" {s2.strip()}")[0]

            prompt_abc = self.prompt_template.from_abc(s1, s2, s3)
            prompt_aba = self.prompt_template.from_aba(s1, s2)

            print(f"{prompt_abc=}")
            baseline_output = self.capture_baseline_output([prompt_abc])
            baseline_probs = torch.softmax(baseline_output.logits, dim=-1)
            top_probs, top_indices = torch.topk(baseline_probs.squeeze(0), k=3)
            assert_shape("top_probs", top_probs, (3,))
            assert_shape("top_indices", top_indices, (3,))
            for i in range(3):
                indices = [int(top_indices[i])]
                print(f"{top_probs[i]:.4f} {self.tokenizer.decode(indices)}")

            logits_prepatched, logits_patched = self.path_patching(
                config,
                baseline_output,
                [prompt_aba],
            )
            assert_shape("logits_prepatched", logits_prepatched, (1, self.V))
            assert_shape("logits_patched", logits_patched, (1, self.V))

            print(f"{prompt_aba=}")
            probs_prepatched = torch.softmax(logits_prepatched, dim=-1)
            assert_shape("probs_prepatched", probs_prepatched, (1, self.V))
            top_probs, top_indices = torch.topk(probs_prepatched.squeeze(0), k=3)
            assert_shape("top_probs", top_probs, (3,))
            assert_shape("top_indices", top_indices, (3,))
            for i in range(3):
                indices = [int(top_indices[i])]
                print(f"{top_probs[i]:.4f} {self.tokenizer.decode(indices)}")

            print(f"{prompt_aba=}")
            probs_patched = torch.softmax(logits_patched, dim=-1)
            assert_shape("probs_patched", probs_patched, (1, self.V))
            top_probs, top_indices = torch.topk(probs_patched.squeeze(0), k=3)
            assert_shape("top_probs", top_probs, (3,))
            assert_shape("top_indices", top_indices, (3,))
            for i in range(3):
                indices = [int(top_indices[i])]
                print(f"{top_probs[i]:.4f} {self.tokenizer.decode(indices)}")

            logits_diff = logits_patched - logits_prepatched
            assert_shape("logits_diff", logits_diff, (1, self.V))

            probs_diff = probs_patched - probs_prepatched
            assert_shape("probs_diff", probs_diff, (1, self.V))

            s1_logit_diff = logits_diff[0, s1_idx].item()
            s2_logit_diff = logits_diff[0, s2_idx].item()
            s1_prob_diff = probs_diff[0, s1_idx].item()
            s2_prob_diff = probs_diff[0, s2_idx].item()

            s1_logits.append(s1_logit_diff)
            s2_logits.append(s2_logit_diff)
            s1_probs.append(s1_prob_diff)
            s2_probs.append(s2_prob_diff)

        mean_s1_logits = torch.tensor(s1_logits, device=self.device).mean().item()
        mean_s2_logits = torch.tensor(s2_logits, device=self.device).mean().item()
        mean_s1_probs = torch.tensor(s1_probs, device=self.device).mean().item()
        mean_s2_probs = torch.tensor(s2_probs, device=self.device).mean().item()

        return DiffLogitsMetrics(
            s1_logit=mean_s1_logits,
            s2_logit=mean_s2_logits,
            s1_prob=mean_s1_probs,
            s2_prob=mean_s2_probs,
        )
