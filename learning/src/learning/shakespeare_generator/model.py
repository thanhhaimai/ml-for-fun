import math
from dataclasses import dataclass
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn, optim

from learning.learner import BatchResult, Config, Learner
from learning.shakespeare_generator.shakespeare_dataset import Sample


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.embedding_size % config.num_heads != 0:
            raise ValueError(
                f"Embedding size {config.embedding_size} must be divisible by number of heads {config.num_heads}"
            )
        self.head_size = config.embedding_size // config.num_heads

        self.query = nn.Linear(config.embedding_size, self.head_size, bias=False)
        self.key = nn.Linear(config.embedding_size, self.head_size, bias=False)
        self.value = nn.Linear(config.embedding_size, self.head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self.tril: torch.Tensor
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.sequence_length, config.sequence_length)),
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
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    config,
                )
                for _ in range(config.num_heads)
            ],
        )
        self.projection = nn.Linear(config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        return: [B, S, E]
        """
        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        # shape: [B, S, H * num_heads] = [B, S, E] since H = E // num_heads
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        assert_shape("output", output, (B, S, E))

        output = self.projection(output)
        assert_shape("output", output, (B, S, E))

        output = self.dropout(output)
        assert_shape("output", output, (B, S, E))

        return output


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(config.embedding_size, config.embedding_size)
        self.gelu = nn.GELU()
        self.projection = nn.Linear(config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, E]
        return: [B, S, E]
        """
        B, S, E = x.shape
        assert_shape("x", x, (B, S, E))

        output = self.linear(x)
        assert_shape("output", output, (B, S, E))

        output = self.gelu(output)
        assert_shape("output", output, (B, S, E))

        output = self.projection(output)
        assert_shape("output", output, (B, S, E))

        output = self.dropout(output)
        assert_shape("output", output, (B, S, E))

        return output


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_size)
        self.heads = MultiHeadAttention(config)

        self.norm2 = nn.LayerNorm(config.embedding_size)
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
        output = x + self.feed_forward(self.norm2(output))
        assert_shape("output", output, (B, S, E))

        return output


class ShakespeareGenerator(nn.Module):
    def __init__(
        self,
        config: Config,
        vocab_size: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.sequence_length = config.sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = config.embedding_size

        # [B, S] of vocab index -> [B, S, E]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embedding_size,
            device=device,
        )

        # [B, S] of positional index -> [B, S, E]
        self.positional_embedding = nn.Embedding(
            num_embeddings=config.sequence_length,
            embedding_dim=config.embedding_size,
            device=device,
        )

        # [B, S, E] -> [B, S, E]
        self.blocks = nn.Sequential(
            Block(config),
            nn.LayerNorm(config.embedding_size),
        )

        # [B, S, E] -> [B, S, V]
        self.linear = nn.Linear(config.embedding_size, vocab_size)

        self.positional_indices: torch.Tensor
        self.register_buffer("positional_indices", torch.arange(config.sequence_length))

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
            next_token_indices = torch.multinomial(probs, num_samples=1)
            assert_shape("next_token_indices", next_token_indices, (B, 1))

            # shape: [B, S + 1]
            original_S = indices.shape[1]
            indices = torch.cat([indices, next_token_indices], dim=-1)
            assert_shape("indices", indices, (B, original_S + 1))

        return indices


@dataclass
class Batch:
    samples: list[Sample]

    @classmethod
    def from_samples(cls, batch: list[Sample]) -> Self:
        return cls(samples=batch)


class ParallelBatchLearner(Learner[Batch]):
    def __init__(
        self,
        model: ShakespeareGenerator,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ):
        super().__init__(model, optimizer, criterion)
        self.model = model
        assert self.criterion.reduction == "sum", "Reduction must be 'sum'"

    def batch_step(self, batch: Batch) -> BatchResult:
        B = len(batch.samples)
        S = batch.samples[0].input.shape[0]
        V = self.model.vocab_size

        # shape: [B, S]
        inputs = torch.stack([sample.input for sample in batch.samples], dim=0)
        assert_shape("inputs", inputs, (B, S))

        # shape: [B, S]
        labels = torch.stack([sample.label for sample in batch.samples], dim=0)
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
