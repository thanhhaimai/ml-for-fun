import glob
import logging
import os
from collections import namedtuple
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NameLabel = namedtuple("NameLabel", ["name", "country_idx"])


class NamesDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        max_countries_count: int | None = None,
        max_names_count: int | None = None,
        transform_input: Callable[[str], str] | None = None,
    ):
        """
        Initializes the NamesDataset by loading names and their associated countries from text files.

        Args:
            data_folder: Path to the folder containing text files, where each file represents a country and contains names.
            max_countries_count: Maximum number of countries to load. If None, all countries are loaded.
            max_names_count: Maximum number of names to load. If None, all names are loaded.
            transform_input: Function to transform names read from the file.

        Raises:
            FileNotFoundError: If the data_folder does not exist.
        """
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"{data_folder=} does not exist.")

        self.max_countries_count = max_countries_count
        self.max_names_count = max_names_count
        self.transform_input = transform_input

        self.names, self.countries, tokens = self.load(data_folder)
        logging.info(f"Total countries loaded: {len(self.countries)}")
        logging.info(f"Total names loaded: {len(self.names)}")
        logging.info(f"Total unique tokens loaded: {len(tokens)}")

        self.index_to_token = tokens
        self.token_to_index = {c: i for i, c in enumerate(tokens)}

        self.names_tensors = []
        self.countries_tensors = []
        for name, country_idx in self.names:
            self.names_tensors.append(self.name_to_tensor(name))
            self.countries_tensors.append(self.country_index_to_tensor(country_idx))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the name and country tensors for the given index.
        """
        return self.names_tensors[idx], self.countries_tensors[idx]

    def load(self, data_folder: str) -> tuple[list[NameLabel], list[str], list[str]]:
        """
        Reads and processes names and their corresponding countries from the specified folder in sorted order.

        Args:
            data_folder: Path to the folder containing text files, where each file represents a country and contains names.

        Returns:
            A list of NameLabel objects (name and country index), a list of country names, and a list of unique tokens.
        """
        names: list[NameLabel] = []
        countries: list[str] = []
        tokens = set()

        all_files = sorted(glob.glob(f"{data_folder}/*.txt"))
        for file_path in all_files:
            if (
                self.max_countries_count is not None
                and len(countries) >= self.max_countries_count
            ):
                logging.info(
                    f"Maximum number of countries {self.max_countries_count} reached. Skipping further countries."
                )
                return names, countries, list(tokens)

            country_name = file_path.split("/")[-1].split(".")[0]
            logging.info(f"Loading names for country: {country_name} from {file_path}")
            countries.append(country_name)

            country_idx = len(countries) - 1
            with open(file_path, "r") as file:
                for line in file:
                    if (
                        self.max_names_count is not None
                        and len(names) >= self.max_names_count
                    ):
                        logging.info(
                            f"Maximum number of names {self.max_names_count} reached. Skipping further names."
                        )
                        return names, countries, list(tokens)

                    name = line.strip()
                    if name:
                        if self.transform_input:
                            name = self.transform_input(name)
                        tokens.update(name)
                        names.append(NameLabel(name=name, country_idx=country_idx))

        return names, countries, list(tokens)

    def to(self, device):
        """
        Moves the dataset tensors to the specified device.
        """
        if self.names_tensors[0].device == device:
            logging.info("Tensors are already on the target device.")
            return

        self.names_tensors = [
            name_tensor.to(device, non_blocking=True)
            for name_tensor in self.names_tensors
        ]
        self.countries_tensors = [
            country_tensor.to(device, non_blocking=True)
            for country_tensor in self.countries_tensors
        ]

    def name_to_tensor(self, name: str) -> torch.Tensor:
        # shape [1, sequence_length, num_classes]
        return (
            F.one_hot(
                torch.tensor([self.token_to_index[c] for c in name]),
                num_classes=len(self.index_to_token),
            )
            .unsqueeze(1)
            .float()
        )

    def tensor_to_name(self, tensor: torch.Tensor) -> str:
        # tensor shape [sequence_length, 1, num_classes]
        indices = tensor.argmax(dim=2).squeeze(1)
        return "".join(self.index_to_token[i] for i in indices)

    def country_index_to_tensor(self, label_index: int) -> torch.Tensor:
        # Return a single integer index as a tensor, compatible with CrossEntropyLoss
        return torch.tensor(label_index, dtype=torch.long)
