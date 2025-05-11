import glob
import unicodedata
from collections import defaultdict
from typing import Self

import torch
import torch.nn.functional as F

START_TOKEN = "."
END_TOKEN = "~"


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
        vocab: list[str],
    ):
        self.country_idx_to_names = names
        self.countries = countries
        self.tokens = vocab

        self.index_to_token = self.tokens
        self.token_to_index = {c: i for i, c in enumerate(self.index_to_token)}

        self.num_classes = len(self.countries)
        self.num_vocab = len(self.tokens)

    @classmethod
    def load(
        cls,
        data_folder: str,
        prefix: str = "",
        suffix: str = "",
        normalize_unicode: bool = False,
    ) -> Self:
        """
        Reads and processes names and their corresponding countries from the specified folder in sorted order.

        Args:
            data_folder: Path to the folder containing text files, where each file represents a country and contains names.
        """
        names: dict[int, list[str]] = defaultdict(list)
        countries: list[str] = []
        vocab = set()

        all_files = sorted(glob.glob(f"{data_folder}/*.txt"))
        for file_path in all_files:
            country_name = file_path.split("/")[-1].split(".")[0]
            countries.append(country_name)

            country_idx = len(countries) - 1
            with open(file_path, "r") as file:
                for line in file:
                    name = line.strip()
                    if not name:
                        continue
                    if normalize_unicode:
                        name = unicode_to_ascii(name)
                    if prefix:
                        name = prefix + name
                    if suffix:
                        name = name + suffix
                    vocab.update(name)
                    names[country_idx].append(name)

        return cls(names, countries, list(sorted(vocab)))

    def t2i(self, name: str) -> list[int]:
        return [self.token_to_index[c] for c in name]

    def i2t(self, indices: list[int]) -> str:
        return "".join(self.index_to_token[i] for i in indices)

    def name_to_one_hot(self, name: str) -> torch.Tensor:
        """
        return: shape [S, 1, V]
        """
        return (
            F.one_hot(
                torch.tensor(self.t2i(name)),
                num_classes=self.num_vocab,
            )
            .float()
            .unsqueeze(1)
        )

    def one_hot_to_name(self, one_hot: torch.Tensor) -> str:
        """
        one_hot: shape [S, V]
        """
        indices = one_hot.argmax(dim=1)
        return self.i2t(indices.tolist())

    def country_index_to_one_hot(self, country_idx: int) -> torch.Tensor:
        """
        return: shape [num_classes]
        """
        return F.one_hot(
            torch.tensor(country_idx),
            num_classes=self.num_classes,
        ).float()
