from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from common.learner import Sample
from data.names_data_source import NamesDataSource


@dataclass
class NameSample(Sample):
    """
    A single non-batched sample of name and country.

    S: sequence_length
    V: num_vocab
    """

    country: str
    name: str


class NamesDataset(Dataset):
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
                        input=self.name_to_one_hot(name),
                        label=torch.tensor(country_idx).unsqueeze(0),
                        country=names_data_source.countries[country_idx],
                        name=name,
                    )
                )

    def name_to_one_hot(self, name: str) -> torch.Tensor:
        """
        return: shape [S, 1, V]
        """
        return (
            F.one_hot(
                torch.tensor(self.names_data_source.t2i(name)),
                num_classes=self.names_data_source.num_vocab,
            )
            .float()
            .unsqueeze(1)
        )

    def one_hot_to_name(self, one_hot: torch.Tensor) -> str:
        """
        one_hot: shape [S, V]
        """
        indices = one_hot.argmax(dim=1)
        return self.names_data_source.i2t(indices.tolist())

    def country_index_to_one_hot(self, country_idx: int) -> torch.Tensor:
        """
        return: shape [num_classes]
        """
        return F.one_hot(
            torch.tensor(country_idx),
            num_classes=self.names_data_source.num_classes,
        ).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> NameSample:
        return self.samples[idx]
