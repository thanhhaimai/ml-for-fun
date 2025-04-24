import glob
import logging
from collections import namedtuple

from torch.utils.data import Dataset

NameLabel = namedtuple("NameLabel", ["name", "country_idx"])


class NamesDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        max_countries_count: int,
        max_names_count: int,
    ):
        """
        Initializes the NamesDataset with a list of names from a text file.

        Args:
            data_folder (str): The path to the folder containing the names file. The file names are processed in sorted order.

        The file within `data_folder` are in format:
        - One name per line
        - File name is the country name
        """
        self.max_countries_count = max_countries_count
        self.max_names_count = max_names_count
        self.names, self.countries = self.load(data_folder)

        logging.info(f"Total names loaded: {len(self.names)}")

    def load(self, data_folder: str) -> tuple[list[NameLabel], list[str]]:
        names: list[NameLabel] = []
        countries: list[str] = []

        all_files = sorted(glob.glob(f"{data_folder}/*.txt"))
        for file_path in all_files:
            if len(countries) > self.max_countries_count:
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
                    if len(names) >= self.max_names_count:
                        logging.info(
                            f"Maximum number of names {self.max_names_count} reached. Skipping further names."
                        )
                        return names, countries

                    name = line.strip()
                    if name:
                        names.append(NameLabel(name=name, country_idx=country_idx))

        return names, countries

    def __len__(self):
        """
        Returns the number of names in the dataset.

        Returns:
            int: The number of names.
        """
        return len(self.names)

    def __getitem__(self, idx) -> tuple[str, str]:
        """
        Returns the name and country index for a given index.

        Args:
            idx (int): The index of the name to retrieve.

        Returns:
            tuple[str, str]: A tuple containing the name and the country name.
        """
        if idx < 0 or idx >= len(self.names):
            raise IndexError(f"{idx=} out of range")

        name, country_idx = self.names[idx]
        return name, self.countries[country_idx]
