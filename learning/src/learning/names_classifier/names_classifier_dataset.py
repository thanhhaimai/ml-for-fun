from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data.names_data_source import NamesDataSource
from data.tokenizer import Tokenizer


@dataclass
class NameSample:
    """
    A single non-batched sample of name and country.

    S: sequence_length
    C: num_classes
    """

    # shape: [S] -- indices of the tokens in the vocabulary
    input: torch.Tensor
    # shape: [1] -- class index
    label: torch.Tensor


class NamesClassifierDataset(Dataset[NameSample]):
    """
    S: sequence_length
    V: num_vocab
    C: num_classes
    """

    def __init__(
        self,
        names_data_source: NamesDataSource,
        tokenizer: Tokenizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.names_data_source = names_data_source
        self.samples: list[NameSample] = []

        for country_idx, names in names_data_source.country_idx_to_names.items():
            label = torch.tensor(
                [country_idx],
                dtype=torch.long,
                device=device,
            )
            for name in names:
                input = torch.tensor(tokenizer.t2i(name), device=device)
                self.samples.append(NameSample(input=input, label=label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> NameSample:
        return self.samples[idx]
