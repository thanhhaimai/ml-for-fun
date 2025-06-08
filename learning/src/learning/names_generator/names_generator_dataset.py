from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.names_data_source import NamesDataSource
from data.tokenizer import Tokenizer


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
        tokenizer: Tokenizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.names_data_source = names_data_source
        self.device = device
        self.samples: list[NameSample] = []

        # shape: [1, V]
        start_token_one_hot = tokenizer.to_one_hot(Tokenizer.START_TOKEN, device=device)
        end_token_idx = tokenizer.token_to_index[Tokenizer.END_TOKEN]

        for country_idx, names in names_data_source.country_idx_to_names.items():
            for name in names:
                # shape: [S+1, V] -- input is padded with start token
                input = torch.cat(
                    [
                        start_token_one_hot,
                        tokenizer.to_one_hot(name, device=device),
                    ],
                    dim=0,
                )
                # shape: [S+1] -- label is padded with end token
                label_indices = [tokenizer.token_to_index[c] for c in name] + [
                    end_token_idx
                ]
                label = torch.tensor(label_indices, dtype=torch.long, device=device)

                # shape: [1, C]
                category = self.country_index_to_one_hot(country_idx).unsqueeze(0)

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

    def country_index_to_one_hot(self, country_idx: int) -> torch.Tensor:
        """
        return: shape [num_classes]
        """
        return F.one_hot(
            torch.tensor(country_idx, device=self.device),
            num_classes=self.names_data_source.num_classes,
        ).float()
