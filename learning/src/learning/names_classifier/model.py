from dataclasses import dataclass
from typing import Literal, Self

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence

from learning.learner import BatchResult, Learner
from learning.names_classifier.names_classifier_dataset import NameSample


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@dataclass
class Config:
    # Training parameters
    batch_size: int
    learning_rate: float
    epochs: int
    patience: int | None
    min_delta: float | None
    device: torch.device

    # The number of tokens
    vocab_size: int
    # The number of countries class
    class_size: int
    # The embedding size
    embedding_size: int
    # The number of hidden units
    hidden_size: int
    # The number of layers
    num_layers: int = 1
    # Whether the network is bidirectional
    bidirectional: bool = False
    # The activation function, either "tanh" or "relu"
    activation: Literal["tanh", "relu"] = "tanh"
    # The dropout rate
    dropout: float = 0.1


class NamesClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.V = config.vocab_size
        self.C = config.class_size

    def predict_topk(
        self, indices: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        indices: shape [S]
        """
        self.eval()
        with torch.no_grad():
            S = len(indices)
            assert_shape("indices", indices, (S,))

            # shape: [1, S]
            padded_inputs = pad_sequence(
                [indices],
                batch_first=True,
                padding_side="left",
            )
            assert_shape("padded_inputs", padded_inputs, (1, S))

            output = self(padded_inputs)
            assert_shape("output", output, (1, self.C))

            # probabilities: [1, C]
            probabilities = torch.softmax(output, dim=-1)
            assert_shape("probabilities", probabilities, (1, self.C))

            # topk_logits: [1, K]
            # indices: [1, K]
            topk_logits, indices = torch.topk(probabilities, k=k, dim=-1)
            assert_shape("topk_logits", topk_logits, (1, k))
            assert_shape("indices", indices, (1, k))

            return topk_logits.squeeze(0), indices.squeeze(0)


class NamesClassifierRNN(NamesClassifier):
    """
    V: vocab_size
    C: class_size
    E: embedding_size
    H: hidden_size
    D: num_layers

    S: sequence_length
    B: batch_size
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.E = config.embedding_size
        self.H = config.hidden_size
        self.D = config.num_layers
        self.bidirectional = config.bidirectional
        if config.bidirectional:
            self.D *= 2

        # embedding: [B, S] -> [B, S, E]
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_size,
            padding_idx=0,
            device=config.device,
        )

        self.ln1 = nn.LayerNorm(
            config.embedding_size,
            device=config.device,
        )

        # rnn: [B, S, E] -> hidden [D, B, H] or [D, B, H * 2]
        self.rnn = nn.RNN(
            batch_first=True,
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            nonlinearity=config.activation,
            dropout=config.dropout,
            device=config.device,
        )

        if self.bidirectional:
            self.ln2 = nn.LayerNorm(
                config.hidden_size * 2,
                device=config.device,
            )
            # fc: [B, H * 2] -> [B, C]
            self.fc = nn.Linear(
                in_features=config.hidden_size * 2,
                out_features=config.class_size,
                device=config.device,
            )
        else:
            self.ln2 = nn.LayerNorm(
                config.hidden_size,
                device=config.device,
            )
            # fc: [B, H] -> [B, C]
            self.fc = nn.Linear(
                in_features=config.hidden_size,
                out_features=config.class_size,
                device=config.device,
            )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, S] -- the indices of the tokens in the vocabulary, by batch, by sequence
        """
        B, S = indices.shape

        # embedding: [B, S, E]
        embedding = self.embedding(indices)
        assert_shape("embedding", embedding, (B, S, self.E))

        # shape: [B, S, E]
        embedding = self.ln1(embedding)
        assert_shape("embedding", embedding, (B, S, self.E))

        # hidden: [D, B, H]
        _rnn_output, hidden = self.rnn(embedding)
        assert_shape("hidden", hidden, (self.D, B, self.H))

        if self.bidirectional:
            # NOTE: basic indexing removes the `D` dimension
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            assert_shape("hidden", hidden, (B, self.H * 2))
            # shape: [B, H * 2]
            hidden = self.ln2(hidden)
            assert_shape("hidden", hidden, (B, self.H * 2))
        else:
            # NOTE: basic indexing removes the `D` dimension
            hidden = hidden[-1]
            assert_shape("hidden", hidden, (B, self.H))
            # shape: [B, H]
            hidden = self.ln2(hidden)
            assert_shape("hidden", hidden, (B, self.H))

        # output: [B, C]
        output = self.fc(hidden)
        return output


class NamesClassifierLSTM(NamesClassifier):
    """
    V: vocab_size
    C: class_size
    E: embedding_size
    H: hidden_size
    D: num_layers

    S: sequence_length
    B: batch_size
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.E = config.embedding_size
        self.H = config.hidden_size
        self.D = config.num_layers
        self.bidirectional = config.bidirectional
        if config.bidirectional:
            self.D *= 2

        # embedding: [B, S] -> [B, S, E]
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_size,
            padding_idx=0,
            device=config.device,
        )

        self.ln1 = nn.LayerNorm(
            config.embedding_size,
            device=config.device,
        )

        # lstm: [S, B, V] -> hidden [D, B, H] or [D, B, H * 2]
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
            device=config.device,
        )

        if self.bidirectional:
            self.ln2 = nn.LayerNorm(
                config.hidden_size * 2,
                device=config.device,
            )
            # fc: [B, H * 2] -> [B, C]
            self.fc = nn.Linear(
                in_features=config.hidden_size * 2,
                out_features=config.class_size,
                device=config.device,
            )
        else:
            self.ln2 = nn.LayerNorm(
                config.hidden_size,
                device=config.device,
            )
            # fc: [B, H] -> [B, C]
            self.fc = nn.Linear(
                in_features=config.hidden_size,
                out_features=config.class_size,
                device=config.device,
            )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, S] -- the indices of the tokens in the vocabulary, by batch, by sequence
        """
        B, S = indices.shape

        # embedding: [B, S, E]
        embedding = self.embedding(indices)
        assert_shape("embedding", embedding, (B, S, self.E))

        # shape: [B, S, E]
        embedding = self.ln1(embedding)
        assert_shape("embedding", embedding, (B, S, self.E))

        # hidden: [D, B, H] or [D, B, H * 2]
        _lstm_output, (hidden, _cell) = self.lstm(embedding)

        if self.bidirectional:
            # NOTE: basic indexing removes the `D` dimension
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            assert_shape("hidden", hidden, (B, self.H * 2))
            # shape: [B, H * 2]
            hidden = self.ln2(hidden)
            assert_shape("hidden", hidden, (B, self.H * 2))
        else:
            # NOTE: basic indexing removes the `D` dimension
            hidden = hidden[-1]
            assert_shape("hidden", hidden, (B, self.H))
            # shape: [B, H]
            hidden = self.ln2(hidden)
            assert_shape("hidden", hidden, (B, self.H))

        # output: [B, C]
        output = self.fc(hidden)
        return output


class NamesClassifierGRU(NamesClassifier):
    """
    V: input_size
    C: class_size
    E: embedding_size
    H: hidden_size
    D: num_layers

    S: sequence_length
    B: batch_size
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.E = config.embedding_size
        self.H = config.hidden_size
        self.D = config.num_layers
        self.bidirectional = config.bidirectional
        if config.bidirectional:
            self.D *= 2

        # embedding: [B, S] -> [B, S, E]
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_size,
            padding_idx=0,
            device=config.device,
        )

        self.ln1 = nn.LayerNorm(
            config.embedding_size,
            device=config.device,
        )

        # gru: [S, B, V] or PackedSequence -> hidden [D, B, H] or [D, B, H * 2]
        self.gru = nn.GRU(
            batch_first=True,
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
            device=config.device,
        )

        if self.bidirectional:
            self.ln2 = nn.LayerNorm(
                config.hidden_size * 2,
                device=config.device,
            )
            # fc: [B, H * 2] -> [B, C]
            self.fc = nn.Linear(
                in_features=config.hidden_size * 2,
                out_features=config.class_size,
                device=config.device,
            )
        else:
            self.ln2 = nn.LayerNorm(
                config.hidden_size,
                device=config.device,
            )
            # fc: [B, H] -> [B, C]
            self.fc = nn.Linear(
                in_features=config.hidden_size,
                out_features=config.class_size,
                device=config.device,
            )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, S] -- the indices of the tokens in the vocabulary, by batch, by sequence
        """
        B, S = indices.shape

        # embedding: [B, S, E]
        embedding = self.embedding(indices)
        assert_shape("embedding", embedding, (B, S, self.E))

        # shape: [B, S, E]
        embedding = self.ln1(embedding)
        assert_shape("embedding", embedding, (B, S, self.E))

        # hidden: [D, B, H] or [D, B, H * 2]
        _gru_output, hidden = self.gru(embedding)

        if self.bidirectional:
            # NOTE: basic indexing removes the `D` dimension
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            assert_shape("hidden", hidden, (B, self.H * 2))
            # shape: [B, H * 2]
            hidden = self.ln2(hidden)
            assert_shape("hidden", hidden, (B, self.H * 2))
        else:
            # NOTE: basic indexing removes the `D` dimension
            hidden = hidden[-1]
            assert_shape("hidden", hidden, (B, self.H))
            # shape: [B, H]
            hidden = self.ln2(hidden)
            assert_shape("hidden", hidden, (B, self.H))

        # output: [B, C]
        output = self.fc(hidden)
        return output


@dataclass
class Batch:
    samples: list[NameSample]

    @classmethod
    def from_samples(cls, batch: list[NameSample]) -> Self:
        return cls(samples=batch)


class ParallelBatchLearner(Learner):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        config: Config,
    ):
        super().__init__(model, optimizer, criterion, config.device)
        self.config = config

    def batch_step(self, batch: Batch) -> BatchResult:
        # Sort samples by input length in descending order
        batch.samples.sort(key=lambda x: len(x.input), reverse=True)
        B = len(batch.samples)
        max_S = batch.samples[0].input.shape[0]

        # sample.input: [S]
        # padded_inputs: [B, max_S]
        padded_inputs = pad_sequence(
            [sample.input for sample in batch.samples],
            batch_first=True,
            padding_side="left",
        )
        assert_shape("padded_inputs", padded_inputs, (B, max_S))

        # shape: [B, C]
        C = self.config.class_size
        outputs = self.model(padded_inputs)
        assert_shape("outputs", outputs, (B, C))

        # sample.label: [1]
        # labels: [B]
        labels = torch.cat([sample.label for sample in batch.samples])
        assert_shape("labels", labels, (B,))
        batch_loss = self.criterion(outputs, labels)

        return BatchResult(
            outputs=outputs,
            labels=labels,
            loss=batch_loss,
            sample_count=len(batch.samples),
        )
