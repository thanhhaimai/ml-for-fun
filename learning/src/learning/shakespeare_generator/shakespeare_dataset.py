from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data.shakespeare_data_source import ShakespeareDataSource
from data.tokenizer import Tokenizer


@dataclass
class Sample:
    """
    A single non-batched sample of a sequence of tokens.

    T: sequence_length (time dimension)
    """

    # Sequence of indices of tokens
    # shape: [T]
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
        device: torch.device = torch.device("cpu"),
    ):
        self.shakespeare_data_source = shakespeare_data_source
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.samples: list[Sample] = []

        if not self.shakespeare_data_source.text:
            print("WARNING: No text found in the data source!")
            return

        indices = torch.tensor(
            data=tokenizer.t2i(self.shakespeare_data_source.text),
            device=device,
        )

        for i in range(len(indices) - self.sequence_length):
            # shape: [T]
            input = indices[i : i + self.sequence_length]
            # shape: [T]
            label = indices[i + 1 : i + self.sequence_length + 1]

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
