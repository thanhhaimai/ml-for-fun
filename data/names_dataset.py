import glob
import logging
from collections import namedtuple

from torch.utils.data import Dataset

NameLabel = namedtuple("NameLabel", ["name", "country_idx"])


class NamesDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        max_countries_count: int | None = None,
        max_names_count: int | None = None,
    ):
        """
        Initializes the NamesDataset by loading names and their associated countries from text files.

        Args:
            data_folder: Path to the folder containing text files, where each file represents a country and contains names.
            max_countries_count: Maximum number of countries to load. If None, all countries are loaded.
            max_names_count: Maximum number of names to load. If None, all names are loaded.
        """
        self.max_countries_count = max_countries_count
        self.max_names_count = max_names_count
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
                        names.append(NameLabel(name=name, country_idx=country_idx))

        return names, countries

    def __len__(self):
        """
        Returns the total number of names in the dataset.
        """
        return len(self.names)

    def __getitem__(self, idx) -> tuple[str, str]:
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
        return name, self.countries[country_idx]
