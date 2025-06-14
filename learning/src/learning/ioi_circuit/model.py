import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import GPT2LMHeadModel

from learning.learner import BatchResult, Learner
from learning.shakespeare_generator.shakespeare_dataset import Sample


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@dataclass
class ModelConfig:
    embedding_size: int
    num_heads: int
    num_blocks: int
    vocab_size: int = 50257
    sequence_length: int = 1024
    feed_forward_expansion_factor: int = 4
    dropout: float = 0.1
    device: torch.device = torch.device("cpu")


@dataclass
class LearnerConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int | None
    min_delta: float | None


class PretrainedName(StrEnum):
    GPT2_SMALL = "openai-community/gpt2"
    GPT2_MEDIUM = "openai-community/gpt2-medium"
    GPT2_LARGE = "openai-community/gpt2-large"
    GPT2_XL = "openai-community/gpt2-xl"


# Pretrained GPT2 configs
PRETRAINED_CONFIG = {
    PretrainedName.GPT2_SMALL: ModelConfig(
        num_blocks=12,
        num_heads=12,
        embedding_size=768,
    ),
    PretrainedName.GPT2_MEDIUM: ModelConfig(
        num_blocks=24,
        num_heads=16,
        embedding_size=1024,
    ),
    PretrainedName.GPT2_LARGE: ModelConfig(
        num_blocks=36,
        num_heads=20,
        embedding_size=1280,
    ),
    PretrainedName.GPT2_XL: ModelConfig(
        num_blocks=48,
        num_heads=25,
        embedding_size=1600,
    ),
}


@dataclass
class HeadId:
    block_idx: int
    head_idx: int


class AttentionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.embedding_size % config.num_heads != 0:
            raise ValueError(
                f"Embedding size {config.embedding_size} must be divisible by number of heads {config.num_heads}"
            )
        self.head_size = config.embedding_size // config.num_heads

        self.query = nn.Linear(
            config.embedding_size,
            self.head_size,
            device=config.device,
        )
        self.key = nn.Linear(
            config.embedding_size,
            self.head_size,
            device=config.device,
        )
        self.value = nn.Linear(
            config.embedding_size,
            self.head_size,
            device=config.device,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.tril: torch.Tensor
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    config.sequence_length,
                    config.sequence_length,
                    device=config.device,
                ),
            ),
        )

        self.should_capture_output = False
        self.use_frozen_output = False
        self.register_buffer("frozen_output", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.should_capture_output and self.use_frozen_output:
            previous_frozen_output = self.frozen_output
            self.frozen_output = self._forward_impl(x)
            return previous_frozen_output
        elif self.use_frozen_output:
            return self.frozen_output
        elif self.should_capture_output:
            self.frozen_output = self._forward_impl(x)
            return self.frozen_output
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        return: [B, S, H] -- NOTE: this is H, not E
        """
        H = self.head_size

        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        # shape: [B, S, H]
        query = self.query(x)
        assert_shape("query", query, (B, S, H))

        # shape: [B, S, H]
        key = self.key(x)
        assert_shape("key", key, (B, S, H))

        # shape: [B, S, H]
        value = self.value(x)
        assert_shape("value", value, (B, S, H))

        # shape: [B, S, H] @ [B, H, S] -> [B, S, S]
        attention = query @ key.transpose(-2, -1) * (self.head_size**-0.5)
        attention = attention.masked_fill(self.tril[:S, :S] == 0, -math.inf)
        attention = F.softmax(attention, dim=-1)
        assert_shape("attention", attention, (B, S, S))

        attention = self.dropout(attention)
        assert_shape("attention", attention, (B, S, S))

        # shape: [B, S, S] @ [B, S, H] -> [B, S, H]
        output = attention @ value
        assert_shape("output", output, (B, S, H))
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads

        self.heads = [AttentionHead(config) for _ in range(config.num_heads)]
        self.heads_module = nn.ModuleList(self.heads)

        self.projection = nn.Linear(
            config.embedding_size,
            config.embedding_size,
            device=config.device,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        # shape: [B, S, H * num_heads] = [B, S, E] since H = E // num_heads
        output = torch.cat([head(x) for head in self.heads_module], dim=-1)
        assert_shape("output", output, (B, S, E))

        output = self.projection(output)
        assert_shape("output", output, (B, S, E))

        output = self.dropout(output)
        assert_shape("output", output, (B, S, E))

        return output


class MultiHeadFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.embedding_size % config.num_heads != 0:
            raise ValueError(
                f"Embedding size {config.embedding_size} must be divisible by number of heads {config.num_heads}"
            )
        self.num_heads = config.num_heads
        self.head_size = config.embedding_size // config.num_heads

        # shape: [B, S, E] -> [B, S, E * 3]
        self.qkv_fc = nn.Linear(
            config.embedding_size, config.embedding_size * 3, device=config.device
        )

        self.projection = nn.Linear(
            config.embedding_size,
            config.embedding_size,
            device=config.device,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.should_capture_output = False
        self.use_frozen_output = False
        self.register_buffer("frozen_output", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.should_capture_output and self.use_frozen_output:
            previous_frozen_output = self.frozen_output
            self.frozen_output = self._forward_impl(x)
            return previous_frozen_output
        elif self.use_frozen_output:
            return self.frozen_output
        elif self.should_capture_output:
            self.frozen_output = self._forward_impl(x)
            return self.frozen_output
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        return: [B, S, E]
        """
        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        # shape: [B, S, E]
        q, k, v = self.qkv_fc(x).split(E, dim=-1)
        assert_shape("q", q, (B, S, E))
        assert_shape("k", k, (B, S, E))
        assert_shape("v", v, (B, S, E))

        # shape: [B, num_heads, S, E // num_heads]
        q = q.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        assert_shape("q", q, (B, self.num_heads, S, self.head_size))

        # shape: [B, num_heads, S, E // num_heads]
        k = k.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        assert_shape("k", k, (B, self.num_heads, S, self.head_size))

        # shape: [B, num_heads, S, E // num_heads]
        v = v.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        assert_shape("v", v, (B, self.num_heads, S, self.head_size))

        # shape: [B, num_heads, S, E // num_heads]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        assert_shape("output", output, (B, self.num_heads, S, self.head_size))

        output = output.transpose(1, 2).contiguous().view(B, S, E)
        assert_shape("output", output, (B, S, E))

        output = self.projection(output)
        assert_shape("output", output, (B, S, E))

        output = self.dropout(output)
        assert_shape("output", output, (B, S, E))

        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.feed_forward_expansion_factor = config.feed_forward_expansion_factor

        self.linear = nn.Linear(
            config.embedding_size,
            config.embedding_size * config.feed_forward_expansion_factor,
            device=config.device,
        )
        self.gelu = nn.GELU(approximate="tanh")
        self.projection = nn.Linear(
            config.embedding_size * config.feed_forward_expansion_factor,
            config.embedding_size,
            device=config.device,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        return: [B, S, E]
        """
        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        output = self.linear(x)
        assert_shape("output", output, (B, S, E * self.feed_forward_expansion_factor))

        output = self.gelu(output)
        assert_shape("output", output, (B, S, E * self.feed_forward_expansion_factor))

        output = self.projection(output)
        assert_shape("output", output, (B, S, E))

        output = self.dropout(output)
        assert_shape("output", output, (B, S, E))

        return output


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_size, device=config.device)
        self.attention = MultiHeadAttention(config)

        self.layer_norm2 = nn.LayerNorm(config.embedding_size, device=config.device)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        return: [B, S, E]
        """
        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        # shape: [B, S, E]
        output = x + self.attention(self.layer_norm1(x))
        assert_shape("output", output, (B, S, E))

        # shape: [B, S, E]
        output = output + self.feed_forward(self.layer_norm2(output))
        assert_shape("output", output, (B, S, E))

        return output


class GPT2(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.sequence_length = config.sequence_length
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.config = config

        # [B, S] of vocab index -> [B, S, E]
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_size,
            device=config.device,
        )

        # [B, S] of positional index -> [B, S, E]
        self.positional_embedding = nn.Embedding(
            num_embeddings=config.sequence_length,
            embedding_dim=config.embedding_size,
            device=config.device,
        )

        self.dropout = nn.Dropout(config.dropout)

        # [B, S, E] -> [B, S, E]
        self.blocks = [Block(config) for _ in range(config.num_blocks)]
        self.blocks_module = nn.ModuleList(self.blocks)

        self.layer_norm = nn.LayerNorm(config.embedding_size, device=config.device)

        # [B, S, E] -> [B, S, V]
        # NOTE: GPT2 uses bias=False for this last linear layer
        self.linear = nn.Linear(
            config.embedding_size,
            config.vocab_size,
            bias=False,
            device=config.device,
        )

        self.positional_indices: torch.Tensor
        self.register_buffer(
            "positional_indices",
            torch.arange(
                config.sequence_length,
                device=config.device,
            ),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, S]
        """
        E = self.embedding_size
        V = self.vocab_size

        B, S = indices.shape
        assert_shape("indices", indices, (B, S))

        # shape: [B, S, E]
        tokens_embedding = self.embedding(indices)
        assert_shape("tokens_embedding", tokens_embedding, (B, S, E))

        # shape: [B, S]
        positional_indices = self.positional_indices[:S].expand(B, -1)
        assert_shape("positional_indices", positional_indices, (B, S))

        # shape: [B, S, E]
        positional_embedding = self.positional_embedding(positional_indices)
        assert_shape("positional_embedding", positional_embedding, (B, S, E))

        # shape: [B, S, E]
        embedding = tokens_embedding + positional_embedding
        assert_shape("embedding", embedding, (B, S, E))

        embedding = self.dropout(embedding)
        assert_shape("embedding", embedding, (B, S, E))

        # shape: [B, S, E]
        for block in self.blocks_module:
            embedding = block(embedding)
            assert_shape("embedding", embedding, (B, S, E))

        embedding = self.layer_norm(embedding)
        assert_shape("embedding", embedding, (B, S, E))

        # shape: [B, S, V]
        output = self.linear(embedding)
        assert_shape("output", output, (B, S, V))
        return output

    def generate(self, indices: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        indices: [B, S] -- the batched sequence of indices
        max_length: number of tokens to generate
        return: [B, S + max_length] -- after generated max_length tokens
        """
        self.eval()
        V = self.vocab_size
        for _ in range(max_length):
            cropped_indices = indices[:, -self.sequence_length :]
            B, S = cropped_indices.shape

            # shape: [B, S, V]
            output = self(cropped_indices)
            assert_shape("output", output, (B, S, V))

            # shape: [B, V]
            last_output = output[:, -1, :]
            assert_shape("last_output", last_output, (B, V))

            # shape: [B, V]
            probs = torch.softmax(last_output, dim=-1)
            assert_shape("probs", probs, (B, V))

            # shape: [B, 1]
            next_token_indices = probs.multinomial(num_samples=1)
            assert_shape("next_token_indices", next_token_indices, (B, 1))

            # shape: [B, S + 1]
            original_S = indices.shape[1]
            indices = torch.cat([indices, next_token_indices], dim=-1)
            assert_shape("indices", indices, (B, original_S + 1))

        return indices

    def set_capture_output_all(self, should_capture_output: bool):
        for block in self.blocks:
            for head in block.attention.heads:
                head.should_capture_output = should_capture_output

    def set_capture_output_heads(
        self, head_ids: list[HeadId], should_capture_output: bool
    ):
        for head_id in head_ids:
            self.blocks[head_id.block_idx].attention.heads[
                head_id.head_idx
            ].should_capture_output = should_capture_output

    def set_use_frozen_output_all(self, use_frozen_output: bool):
        for block in self.blocks:
            for head in block.attention.heads:
                head.use_frozen_output = use_frozen_output

    def set_use_frozen_output_block(self, block_idx: int, use_frozen_output: bool):
        for head in self.blocks[block_idx].attention.heads:
            head.use_frozen_output = use_frozen_output

    def set_use_frozen_output_heads(
        self, head_ids: list[HeadId], use_frozen_output: bool
    ):
        for head_id in head_ids:
            self.blocks[head_id.block_idx].attention.heads[
                head_id.head_idx
            ].use_frozen_output = use_frozen_output

    @classmethod
    def from_pretrained(
        cls, pretrained_name: PretrainedName, device: torch.device
    ) -> tuple[Self, GPT2LMHeadModel]:
        pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_name)
        pretrained_state = pretrained_model.state_dict()

        pretrained_config = PRETRAINED_CONFIG[pretrained_name]
        pretrained_config.device = device
        model = cls(config=pretrained_config)
        model_state = model.state_dict()

        model_state["embedding.weight"] = pretrained_state["transformer.wte.weight"]
        model_state["positional_embedding.weight"] = pretrained_state[
            "transformer.wpe.weight"
        ]

        # NOTE: on HF, the Linear layers are modeled as Conv1D layers
        # Conv1D shape: [n_in, n_out]
        # Linear shape: [n_out, n_in]
        # So we need to transpose the weights to match our Linear layer
        for i in range(pretrained_config.num_blocks):
            model_state[f"blocks_module.{i}.layer_norm1.weight"] = pretrained_state[
                f"transformer.h.{i}.ln_1.weight"
            ]
            model_state[f"blocks_module.{i}.layer_norm1.bias"] = pretrained_state[
                f"transformer.h.{i}.ln_1.bias"
            ]

            E = pretrained_config.embedding_size
            head_size = E // pretrained_config.num_heads

            # HF c_attn weight: [E, 3*E] in Conv1D format
            # After transpose: [3*E, E] for Linear format
            # The 3*E dimension contains [Q_all, K_all, V_all] concatenated
            # with Q_all shape [E, E],
            #      K_all shape [E, E],
            #      V_all shape [E, E]
            c_attn_weight = pretrained_state[f"transformer.h.{i}.attn.c_attn.weight"].T
            assert_shape("c_attn_weight", c_attn_weight, (3 * E, E))

            c_attn_bias = pretrained_state[f"transformer.h.{i}.attn.c_attn.bias"]
            assert_shape("c_attn_bias", c_attn_bias, (3 * E,))

            # Split into Q, K, V blocks: each [E, E]
            q_weight, k_weight, v_weight = c_attn_weight.split(E, dim=0)
            q_bias, k_bias, v_bias = c_attn_bias.split(E, dim=0)

            # Reshape Q, K, V weights to separate heads
            # HF stores weights as [E, E] where E = num_heads * head_size
            # We need to reshape to [num_heads, head_size, E] then extract each head
            q_weight = q_weight.view(pretrained_config.num_heads, head_size, E)
            k_weight = k_weight.view(pretrained_config.num_heads, head_size, E)
            v_weight = v_weight.view(pretrained_config.num_heads, head_size, E)

            q_bias = q_bias.view(pretrained_config.num_heads, head_size)
            k_bias = k_bias.view(pretrained_config.num_heads, head_size)
            v_bias = v_bias.view(pretrained_config.num_heads, head_size)

            # Each head gets its corresponding slice
            for head_idx in range(pretrained_config.num_heads):
                model_state[
                    f"blocks_module.{i}.attention.heads_module.{head_idx}.query.weight"
                ] = q_weight[head_idx]
                model_state[
                    f"blocks_module.{i}.attention.heads_module.{head_idx}.query.bias"
                ] = q_bias[head_idx]
                model_state[
                    f"blocks_module.{i}.attention.heads_module.{head_idx}.key.weight"
                ] = k_weight[head_idx]
                model_state[
                    f"blocks_module.{i}.attention.heads_module.{head_idx}.key.bias"
                ] = k_bias[head_idx]
                model_state[
                    f"blocks_module.{i}.attention.heads_module.{head_idx}.value.weight"
                ] = v_weight[head_idx]
                model_state[
                    f"blocks_module.{i}.attention.heads_module.{head_idx}.value.bias"
                ] = v_bias[head_idx]

            model_state[f"blocks_module.{i}.attention.projection.weight"] = (
                pretrained_state[f"transformer.h.{i}.attn.c_proj.weight"].T
            )
            model_state[f"blocks_module.{i}.attention.projection.bias"] = (
                pretrained_state[f"transformer.h.{i}.attn.c_proj.bias"]
            )
            model_state[f"blocks_module.{i}.layer_norm2.weight"] = pretrained_state[
                f"transformer.h.{i}.ln_2.weight"
            ]
            model_state[f"blocks_module.{i}.layer_norm2.bias"] = pretrained_state[
                f"transformer.h.{i}.ln_2.bias"
            ]
            model_state[f"blocks_module.{i}.feed_forward.linear.weight"] = (
                pretrained_state[f"transformer.h.{i}.mlp.c_fc.weight"].T
            )
            model_state[f"blocks_module.{i}.feed_forward.linear.bias"] = (
                pretrained_state[f"transformer.h.{i}.mlp.c_fc.bias"]
            )
            model_state[f"blocks_module.{i}.feed_forward.projection.weight"] = (
                pretrained_state[f"transformer.h.{i}.mlp.c_proj.weight"].T
            )
            model_state[f"blocks_module.{i}.feed_forward.projection.bias"] = (
                pretrained_state[f"transformer.h.{i}.mlp.c_proj.bias"]
            )

        model_state["layer_norm.weight"] = pretrained_state["transformer.ln_f.weight"]
        model_state["layer_norm.bias"] = pretrained_state["transformer.ln_f.bias"]
        model_state["linear.weight"] = pretrained_state["lm_head.weight"]

        model.load_state_dict(model_state)
        return model, pretrained_model


@dataclass
class Batch:
    inputs: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def from_samples(cls, samples: list[Sample]) -> Self:
        B = len(samples)
        S = samples[0].input.shape[0]

        # shape: [B, S] - Keep on CPU for multiprocessing compatibility
        inputs = torch.stack([sample.input for sample in samples], dim=0)
        assert_shape("inputs", inputs, (B, S))

        # shape: [B, S] - Keep on CPU for multiprocessing compatibility
        labels = torch.stack([sample.label for sample in samples], dim=0)
        assert_shape("labels", labels, (B, S))

        return cls(inputs=inputs, labels=labels)

    def pin_memory(self) -> Self:
        self.inputs.pin_memory()
        self.labels.pin_memory()
        return self


class BatchLearner(Learner[Batch]):
    def __init__(
        self,
        model: GPT2,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        super().__init__(model, optimizer, criterion, device)
        self.model = model
        assert self.criterion.reduction == "sum", "Reduction must be 'sum'"

    def batch_step(self, batch: Batch) -> BatchResult:
        B, S = batch.inputs.shape
        V = self.model.vocab_size

        # Move tensors to GPU in main process (not in workers)
        inputs = batch.inputs.to(self.device, non_blocking=True)
        labels = batch.labels.to(self.device, non_blocking=True)
        assert_shape("inputs", inputs, (B, S))
        assert_shape("labels", labels, (B, S))

        # shape: [B, S, V]
        outputs = self.model(inputs)
        assert_shape("outputs", outputs, (B, S, V))

        # `sum` mode
        batch_loss = self.criterion(outputs.view(B * S, V), labels.view(B * S))

        return BatchResult(
            outputs=outputs,
            labels=labels,
            loss=batch_loss,
            sample_count=B * S,
        )
