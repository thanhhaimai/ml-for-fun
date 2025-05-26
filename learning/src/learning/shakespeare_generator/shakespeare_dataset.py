from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data.shakespeare_data_source import ShakespeareDataSource
from data.tokenizer import Tokenizer


@dataclass
class Sample:
    """
    A single non-batched sample of a sequence of characters.

    T: sequence_length (time dimension)
    V: num_vocab
    """

    # A sequence of characters is encoded as a one-hot vector
    # shape: [T, V]
    input: torch.Tensor

    # For each sequence in the input, the label is the index of the next token
    # shape: [T]
    label: torch.Tensor


class ShakespeareDataset(Dataset[Sample]):
    """
    T: sequence_length (time dimension)
    V: num_vocab
    """

    def __init__(
        self,
        shakespeare_data_source: ShakespeareDataSource,
        tokenizer: Tokenizer,
        sequence_length: int,
    ):
        self.shakespeare_data_source = shakespeare_data_source
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.samples: list[Sample] = []

        if not self.shakespeare_data_source.text:
            return

        indices = tokenizer.t2i(self.shakespeare_data_source.text)
        # shape: [N, V], with N = len(text)
        one_hot_text = F.one_hot(
            torch.tensor(indices),
            num_classes=tokenizer.vocab_size,
        ).to(torch.float32)

        for i in range(len(one_hot_text) - self.sequence_length):
            # shape: [T, V]
            input = one_hot_text[i : i + self.sequence_length]
            # shape: [T]
            label = torch.tensor(indices[i + 1 : i + self.sequence_length + 1])

            self.samples.append(
                Sample(
                    input=input,
                    label=label,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Sample:
        return self.samples[idx]
