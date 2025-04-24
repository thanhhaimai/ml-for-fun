import glob
import logging
import os
from collections import namedtuple
from collections.abc import Callable
from typing import Generic, TypeVar

from torch.utils.data import Dataset

NameLabel = namedtuple("NameLabel", ["name", "country_idx"])

OutputType = TypeVar("OutputType")
LabelType = TypeVar("LabelType")


class NamesDataset(Dataset, Generic[OutputType, LabelType]):
    def __init__(
        self,
        data_folder: str,
        max_countries_count: int | None = None,
        max_names_count: int | None = None,
        transform_input: Callable[[str], str] | None = None,
        transform_output: Callable[[str], OutputType] | None = None,
        transform_label: Callable[[str], LabelType] | None = None,
    ):
        """
        Initializes the NamesDataset by loading names and their associated countries from text files.

        Args:
            data_folder: Path to the folder containing text files, where each file represents a country and contains names.
            max_countries_count: Maximum number of countries to load. If None, all countries are loaded.
            max_names_count: Maximum number of names to load. If None, all names are loaded.
            transform_input: Function to transform names read from the file.
            transform_output: Function to transform the returned name item.
            transform_label: Function to transform the returned country item.

        Raises:
            FileNotFoundError: If the data_folder does not exist.
        """
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"{data_folder=} does not exist.")

        self.max_countries_count = max_countries_count
        self.max_names_count = max_names_count
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.transform_label = transform_label
        self.names, self.countries = self.load(data_folder)

        logging.info(f"Total names loaded: {len(self.names)}")

    def load(self, data_folder: str) -> tuple[list[NameLabel], list[str]]:
        """
        Reads and processes names and their corresponding countries from the specified folder in sorted order.

        Args:
            data_folder: Path to the folder containing text files, where each file represents a country and contains names.

        Returns:
            A list of NameLabel objects (name and country index) and a list of country names.
        """
        names: list[NameLabel] = []
        countries: list[str] = []

        all_files = sorted(glob.glob(f"{data_folder}/*.txt"))
        for file_path in all_files:
            if (
                self.max_countries_count is not None
                and len(countries) >= self.max_countries_count
            ):
                logging.info(
                    f"Maximum number of countries {self.max_countries_count} reached. Skipping further countries."
                )
                return names, countries

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
                        return names, countries

                    name = line.strip()
                    if name:
                        if self.transform_input:
                            name = self.transform_input(name)
                        names.append(NameLabel(name=name, country_idx=country_idx))

        return names, countries

    def __len__(self):
        """
        Returns the total number of names in the dataset.
        """
        return len(self.names)

    def __getitem__(self, idx) -> tuple[OutputType, LabelType]:
        """
        Retrieves the name and its associated country for a given index.

        Args:
            idx (int): Index of the name to retrieve.

        Returns:
            The name and the corresponding country name.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self.names):
            raise IndexError(f"{idx=} out of range")

        name, country_idx = self.names[idx]
        if self.transform_output:
            name = self.transform_output(name)
        country = self.countries[country_idx]
        if self.transform_label:
            country = self.transform_label(country)
        return name, country
