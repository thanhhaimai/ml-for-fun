from dataclasses import dataclass

import tiktoken
import torch

from learning.gpt2.data_sources import NameSample
from learning.gpt2.metrics import ProbsMetrics
from learning.gpt2.model import GPT2, HeadId
from learning.gpt2.prompts import PromptBatch, PromptTemplate


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@dataclass
class TopKProbsResult:
    # shape: [k]
    top_probs: torch.Tensor
    # shape: [k]
    top_indices: torch.Tensor

    def print(self, tokenizer: tiktoken.Encoding):
        for i in range(len(self.top_probs)):
            indices = [int(self.top_indices[i])]
            print(f"{self.top_probs[i]:.2f} {tokenizer.decode(indices)}")


@dataclass
class CapturedOutput:
    # Mean of the head outputs
    # indexed by block_idx and head_idx
    # shape: [num_blocks, num_heads, B, S, H]
    head_outputs: list[list[torch.Tensor]]

    # Mean of the final logits
    # shape: [B, V]
    logits: torch.Tensor


@dataclass
class PathPatchingConfig:
    start_head: HeadId
    end_heads: list[HeadId]


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
        self.model.set_capture_output_all(True)
        self.model.set_use_frozen_output_all(False)

        logits_abc = self.forward(prompts_abc)
        assert_shape("logits_abc", logits_abc, (B, self.V))

        results = []
        for block_idx in range(self.model.config.num_blocks):
            block_results = []
            for head_idx in range(self.model.config.num_heads):
                head_output = (
                    self.model.blocks[block_idx].attention.heads[head_idx].frozen_output
                )
                S = head_output.shape[1]
                assert_shape("head_output", head_output, (B, S, self.H))

                block_results.append(head_output)
            results.append(block_results)

        return CapturedOutput(
            head_outputs=results,
            logits=logits_abc,
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
        assert_shape("baseline", baseline, (B, S, self.H))

        # ================================
        # Phase B: capture the ABB output (prepatched)
        # ================================
        self.model.set_capture_output_all(True)
        self.model.set_use_frozen_output_all(False)
        logits_prepatched = self.forward(prompts_abb)
        assert_shape("logits_prepatched", logits_prepatched, (B, self.V))

        # ================================
        # Phase C: patch the start_head using the ABC start_head output
        # ================================
        # Effectively, we're knocking out the start_head output
        # If the head is not important, the patched output should be close to the ABB output
        # NOTE: the patched end_heads output is not used within this phase (it is used in Phase D)
        self.model.blocks[config.start_head.block_idx].attention.heads[
            config.start_head.head_idx
        ].frozen_output = baseline

        # Run the model using the frozen outputs
        # But also capture all end_heads outputs
        # NOTE: the captured end_heads outputs are not used within this phase (they are used in Phase D)
        self.model.set_capture_output_all(False)
        self.model.set_capture_output_heads(config.end_heads, True)
        self.model.set_use_frozen_output_all(True)
        logits_patched = self.forward(prompts_abb)
        assert_shape("logits_patched", logits_patched, (B, self.V))

        # ================================
        # Phase D: Forward with the end_heads frozen output
        # ================================
        if config.end_heads:
            self.model.set_capture_output_all(False)

            # any block after this layer will not use the frozen output except for the end_heads
            min_block_idx = min(head.block_idx for head in config.end_heads)

            # - First, mark all heads to use frozen output
            self.model.set_use_frozen_output_all(True)

            # - Second, mark all blocks after the min_block_idx to not use frozen output
            #   This means the network is split into two parts: the first part uses the frozen output, the second part does not
            for block_idx in range(min_block_idx + 1, self.model.config.num_blocks):
                self.model.set_use_frozen_output_block(block_idx, False)

            # - Third, mark the end_heads to use the frozen output
            #   The end_heads are the only heads in the second part that use the frozen output
            self.model.set_use_frozen_output_heads(config.end_heads, True)

            logits_patched = self.forward(prompts_abb)
            assert_shape("patched_probs", logits_patched, (B, self.V))

        return logits_prepatched, logits_patched

    def analyze_head(
        self,
        config: PathPatchingConfig,
        baseline_output: CapturedOutput,
        batch: PromptBatch,
    ) -> ProbsMetrics:
        """
        Analyze a set of paths specified by `config`

        The mean of `baseline_output` is used as the knockout baseline for the `config.start_head`.

        The `baseline_output` is generated using a batch of random (s1, s2, s3) names.
        By using the mean of the baseline output, we are effectively averaging out all the "noise" related to names.

        Then, we patch the `config.start_head` output using this baseline. That effectively knocks out the `config.start_head`.
        In other words, the contribution of the `config.start_head` to the output is removed.

        Returns the metrics between the original abb output and the patched output.
        """
        B = len(batch.prompts)

        # This version of `analyze_head` average out the baseline before patching
        baseline = baseline_output.head_outputs[config.start_head.block_idx][
            config.start_head.head_idx
        ]
        S = baseline.shape[1]
        assert_shape("baseline", baseline, (B, S, self.H))
        baseline = baseline.mean(dim=0, keepdim=True)
        assert_shape("baseline", baseline, (1, S, self.H))
        baseline = baseline.expand(B, S, self.H)
        assert_shape("baseline", baseline, (B, S, self.H))

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
            s3_indices=batch.s3_indices,
        )

    def analyze_head_pairwise(
        self, config: PathPatchingConfig, name_samples: list[NameSample]
    ) -> ProbsMetrics:
        """
        Analyze a set of paths specified by `config`

        This version doesn't average out the baseline output.
        Which means for each sample in the batch, the baseline use (s1, s2, s3) and the corresponding patched version use (s1, s2)
        When we patch the `config.start_head` output, the other signal related to the meaning of (s1, s2, s3) is still there.

        This is different from `analyze_head`; it doesn't effectively knock out the `config.start_head`.
        We're comparing how much extra signal the `start_head` is getting from from replacing s2 with s3.

        Returns the metrics between the original abb output and the patched output.
        """
        B = len(name_samples)

        prompts_abc = []
        prompts_abb = []
        s1_indices = []
        s2_indices = []
        s3_indices = []
        for name_sample in name_samples:
            s1, s2, s3 = name_sample.names_with_space
            prompts_abc.append(self.prompt_template.from_abc(s1, s2, s3))
            prompts_abb.append(self.prompt_template.from_abb(s1, s2))
            s1_indices.append(name_sample.indices[0])
            s2_indices.append(name_sample.indices[1])
            s3_indices.append(name_sample.indices[2])

        # This version of `analyze_head_pairwise` keeps the baseline output matching 1:1 to the patched output
        baseline_output = self.capture_baseline_output(prompts_abc)

        logits_prepatched, logits_patched = self.path_patching(
            config,
            baseline_output,
            prompts_abb,
        )
        assert_shape("logits_prepatched", logits_prepatched, (B, self.V))
        assert_shape("logits_patched", logits_patched, (B, self.V))

        return ProbsMetrics.from_logits(
            original_logits=logits_prepatched,
            patched_logits=logits_patched,
            s1_indices=torch.tensor(s1_indices, device=self.device),
            s2_indices=torch.tensor(s2_indices, device=self.device),
            s3_indices=torch.tensor(s3_indices, device=self.device),
        )
