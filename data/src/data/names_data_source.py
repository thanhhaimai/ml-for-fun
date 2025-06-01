import glob
import unicodedata
from collections import Counter, defaultdict
from typing import Self

import matplotlib.pyplot as plt

from data.tokenizer import Tokenizer


def unicode_to_ascii(s: str) -> str:
    """
    Turn a Unicode string to plain ASCII.

    1) Decomposes characters into their canonical base characters and combining diacritical marks using "NFD" form.
        For example, the character "é" (e with acute accent) would be decomposed into:
        - "e" (the base character)
        - "´" (the combining acute accent).

    2) Filters out "Mn" characters. "Mn" stands for "Mark, Nonspacing".
        These are characters like accents, umlauts, etc., that modify a preceding character
        but do not take up space on their own (e.g., the "´" from the "é" example).
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


class NamesDataSource:
    def __init__(
        self,
        names: dict[int, list[str]],
        countries: list[str],
        tokenizer: Tokenizer,
    ):
        self.country_idx_to_names = names
        self.countries = countries
        self.tokenizer = tokenizer
        self.num_classes = len(self.countries)

    @classmethod
    def load(
        cls,
        data_folder: str,
        tokenizer: Tokenizer,
        normalize_unicode: bool = False,
        lowercase: bool = False,
    ) -> Self:
        """
        Reads and processes names and their corresponding countries from the specified folder in sorted order.

        Args:
            data_folder: Path to the folder containing text files, where each file represents a country and contains names.
        """
        data: dict[int, list[str]] = defaultdict(list)
        all_countries: list[str] = []

        all_files = sorted(glob.glob(f"{data_folder}/*.txt"))
        vocab = set()
        for file_path in all_files:
            country_name = file_path.split("/")[-1].split(".")[0]
            all_countries.append(country_name)

            country_idx = len(all_countries) - 1
            with open(file_path, "r") as file:
                names = set()
                for line in file:
                    name = line.strip()
                    if not name:
                        continue
                    if normalize_unicode:
                        name = unicode_to_ascii(name)
                    if lowercase:
                        name = name.lower()
                    vocab.update(name)
                    names.add(name)

            data[country_idx].extend(sorted(names))

        tokenizer.load(vocab)
        return cls(data, all_countries, tokenizer)

    @property
    def class_frequency(self) -> list[float]:
        return [len(names) for names in self.country_idx_to_names.values()]

    @property
    def token_frequency(self) -> list[float]:
        counter = Counter()
        for names in self.country_idx_to_names.values():
            for name in names:
                counter.update(name)
        return [counter[token] for token in self.tokenizer.index_to_token]

    def plot_class_frequency(self, fig_size: tuple[int, int]):
        f, ax = plt.subplots(figsize=fig_size)
        ax.set_title("Class Frequency")
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency")
        ax.set_xticks(range(len(self.countries)))
        ax.set_xticklabels(self.countries, rotation=45, ha="right")
        ax.bar(range(len(self.countries)), self.class_frequency)
        plt.show()

    def plot_token_frequency(self, fig_size: tuple[int, int]):
        f, ax = plt.subplots(figsize=fig_size)
        ax.set_title("Token Frequency")
        ax.set_xlabel("Token")
        ax.set_ylabel("Frequency")
        ax.set_xticks(range(len(self.tokenizer.index_to_token)))
        ax.set_xticklabels(self.tokenizer.index_to_token, rotation=45, ha="right")
        ax.bar(range(len(self.tokenizer.index_to_token)), self.token_frequency)
        plt.show()
