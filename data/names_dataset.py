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
        Initializes the NamesDataset with a list of names from text files.

        Args:
            data_folder (str): The path to the folder containing the names files.
              Each file represents a country and contains one name per line.
            max_countries_count (int): The maximum number of countries to load.
            max_names_count (int): The maximum number of names to load.

        The files within `data_folder` should follow this format:
        - Each file is named after a country (e.g., `English.txt`).
        - Each line in the file contains a single name.
        """
        self.max_countries_count = max_countries_count
        self.max_names_count = max_names_count
        self.names, self.countries = self.load(data_folder)

        logging.info(f"Total names loaded: {len(self.names)}")

    def load(self, data_folder: str) -> tuple[list[NameLabel], list[str]]:
        """
        Loads names and their corresponding countries from the specified folder.

        Args:
            data_folder (str): The path to the folder containing the names files.
            Each file represents a country and contains one name per line.

        Returns:
            tuple[list[NameLabel], list[str]]: A tuple containing a list of NameLabel objects (name and country index)
            and a list of country names.

        Notes:
            - Files are processed in sorted order by filename.
            - Stops loading if the maximum number of countries or names is reached.
        """
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
            int: The total number of names loaded into the dataset.
        """
        return len(self.names)

    def __getitem__(self, idx) -> tuple[str, str]:
        """
        Retrieves the name and corresponding country for a given index.

        Args:
            idx (int): The index of the name to retrieve.

        Returns:
            tuple[str, str]: A tuple containing the name and the country name.

        Raises:
            IndexError: If the provided index is out of range.
        """
        if idx < 0 or idx >= len(self.names):
            raise IndexError(f"{idx=} out of range")

        name, country_idx = self.names[idx]
        return name, self.countries[country_idx]
