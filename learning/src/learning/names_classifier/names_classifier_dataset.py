from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data.names_data_source import NamesDataSource


@dataclass
class NameSample:
    """
    A single non-batched sample of name and country.

    S: sequence_length
    C: num_classes
    """

    # shape: [S, V] -- one-hot encoded sequence
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
    ):
        self.names_data_source = names_data_source
        self.samples: list[NameSample] = []

        for country_idx, names in names_data_source.country_idx_to_names.items():
            for name in names:
                self.samples.append(
                    NameSample(
                        input=self.names_data_source.name_to_one_hot(name),
                        label=torch.tensor(country_idx).unsqueeze(0),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> NameSample:
        return self.samples[idx]
