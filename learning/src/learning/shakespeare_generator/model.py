import math
from dataclasses import dataclass
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn, optim

from learning.learner import BatchResult, Learner
from learning.shakespeare_generator.shakespeare_dataset import Sample


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@dataclass
class Config:
    batch_size: int
    sequence_length: int
    embedding_size: int
    num_heads: int
    num_blocks: int
    epochs: int
    dropout: float
    learning_rate: float
    patience: int | None
    min_delta: float | None
    device: torch.device


class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.embedding_size % config.num_heads != 0:
            raise ValueError(
                f"Embedding size {config.embedding_size} must be divisible by number of heads {config.num_heads}"
            )
        self.head_size = config.embedding_size // config.num_heads

        self.query = nn.Linear(
            config.embedding_size,
            self.head_size,
            bias=False,
            device=config.device,
        )
        self.key = nn.Linear(
            config.embedding_size,
            self.head_size,
            bias=False,
            device=config.device,
        )
        self.value = nn.Linear(
            config.embedding_size,
            self.head_size,
            bias=False,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, config: Config):
        super().__init__()
        if config.embedding_size % config.num_heads != 0:
            raise ValueError(
                f"Embedding size {config.embedding_size} must be divisible by number of heads {config.num_heads}"
            )
        self.num_heads = config.num_heads
        self.head_size = config.embedding_size // config.num_heads

        # shape: [B, S, E] -> [B, S, E * 3]
        self.qkv_fc = nn.Linear(
            config.embedding_size,
            config.embedding_size * 3,
            device=config.device,
        )

        self.projection = nn.Linear(
            config.embedding_size,
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

        # shape: [B, S, E]
        q, k, v = self.qkv_fc(x).split(E, dim=-1)
        assert_shape("q", q, (B, S, E))
        assert_shape("k", k, (B, S, E))
        assert_shape("v", v, (B, S, E))

        # shape: [B, H, S, E // H]
        q = q.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        assert_shape("q", q, (B, self.num_heads, S, self.head_size))

        # shape: [B, H, S, E // H]
        k = k.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        assert_shape("k", k, (B, self.num_heads, S, self.head_size))

        # shape: [B, H, S, E // H]
        v = v.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        assert_shape("v", v, (B, self.num_heads, S, self.head_size))

        # shape: [B, H, S, E // H]
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert_shape("output", output, (B, self.num_heads, S, self.head_size))

        output = output.transpose(1, 2).contiguous().view(B, S, E)
        assert_shape("output", output, (B, S, E))

        output = self.projection(output)
        assert_shape("output", output, (B, S, E))

        output = self.dropout(output)
        assert_shape("output", output, (B, S, E))

        return output


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(
            config.embedding_size,
            config.embedding_size * 2,
            device=config.device,
        )
        self.gelu = nn.GELU()
        self.projection = nn.Linear(
            config.embedding_size * 2,
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
        assert_shape("output", output, (B, S, E * 2))

        output = self.gelu(output)
        assert_shape("output", output, (B, S, E * 2))

        output = self.projection(output)
        assert_shape("output", output, (B, S, E))

        output = self.dropout(output)
        assert_shape("output", output, (B, S, E))

        return output


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.embedding_size,
            device=config.device,
        )
        self.heads = MultiHeadAttention(config)

        self.norm2 = nn.LayerNorm(
            config.embedding_size,
            device=config.device,
        )
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        return: [B, S, E]
        """
        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        # shape: [B, S, E]
        output = x + self.heads(self.norm1(x))
        assert_shape("output", output, (B, S, E))

        # shape: [B, S, E]
        output = output + self.feed_forward(self.norm2(output))
        assert_shape("output", output, (B, S, E))

        return output


class ShakespeareGenerator(nn.Module):
    def __init__(
        self,
        config: Config,
        vocab_size: int,
    ):
        super().__init__()
        self.sequence_length = config.sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = config.embedding_size

        # [B, S] of vocab index -> [B, S, E]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embedding_size,
            device=config.device,
        )

        # [B, S] of positional index -> [B, S, E]
        self.positional_embedding = nn.Embedding(
            num_embeddings=config.sequence_length,
            embedding_dim=config.embedding_size,
            device=config.device,
        )

        # [B, S, E] -> [B, S, E]
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)],
            nn.LayerNorm(
                config.embedding_size,
                device=config.device,
            ),
        )

        # [B, S, E] -> [B, S, V]
        self.linear = nn.Linear(
            config.embedding_size,
            vocab_size,
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

        # shape: [B, S, E]
        block_output = self.blocks(embedding)
        assert_shape("block_output", block_output, (B, S, E))

        # shape: [B, S, V]
        output = self.linear(block_output)
        assert_shape("output", output, (B, S, V))
        return output

    def generate(self, indices: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        indices: [B, S] -- the batched sequence of indices
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


class ParallelBatchLearner(Learner[Batch]):
    def __init__(
        self,
        model: ShakespeareGenerator,
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
