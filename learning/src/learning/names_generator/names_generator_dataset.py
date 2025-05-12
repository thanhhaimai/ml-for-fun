from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data.names_data_source import NamesDataSource


@dataclass
class NameSample:
    """
    A single non-batched sample of name and country.

    S: sequence_length
    V: num_vocab
    C: num_classes
    """

    # A sequence of characters is encoded as a one-hot vector
    # shape: [S, V]
    input: torch.Tensor

    # For each character in the input, the label is the index of the next character
    # shape: [S]
    label: torch.Tensor

    # The whole sequence is from the same country, encoded as a one-hot vector
    # shape: [1, C]
    category: torch.Tensor


class NamesGeneratorDataset(Dataset[NameSample]):
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
                # shape: [S, 1, V]
                name_one_hot = self.names_data_source.name_to_one_hot(name)
                # shape: [S, V]
                name_one_hot.squeeze_(dim=1)

                # shape: [S, V]
                input = name_one_hot[:-1]
                # shape: [S]
                label = torch.tensor(
                    [self.names_data_source.t2i(c) for c in name[1:]], dtype=torch.long
                ).squeeze(1)
                # shape: [1, C]
                category = names_data_source.country_index_to_one_hot(
                    country_idx
                ).unsqueeze(0)

                self.samples.append(
                    NameSample(
                        category=category,
                        input=input,
                        label=label,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> NameSample:
        return self.samples[idx]
