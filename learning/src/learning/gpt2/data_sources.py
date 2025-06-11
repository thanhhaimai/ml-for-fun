import random
from dataclasses import dataclass
from typing import Self

import tiktoken


@dataclass
class NameSample:
    names_with_space: list[str]
    indices: list[int]


class NamesDataSource:
    """
    Data source for popular names that ensures all names (with space) are single tokens.

    All names are stored with a space prefix.
    """

    def __init__(
        self,
        names_with_space: list[str],
        indices: list[int],
    ):
        assert len(names_with_space) == len(indices)
        self.names_with_space = names_with_space
        self.indices = indices

    @classmethod
    def load(
        cls,
        file_path: str,
        tokenizer: tiktoken.Encoding,
    ) -> Self:
        """
        Load names from a file and filter to single-token names only.

        1. Reads all unique names from the file, removing duplicates and whitespace
        3. Filters names with space prefix to single tokens only
        """
        with open(file_path, "r") as file:
            # Read all unique names
            names_no_space = list(set(line.strip() for line in file if line.strip()))

            # Filter out names (with space) that are not single tokens
            names_with_space = [f" {name}" for name in names_no_space]
            indices_with_space = tokenizer.encode_batch(names_with_space)
            indices_with_space = [
                indices for indices in indices_with_space if len(indices) == 1
            ]
            names_with_space = tokenizer.decode_batch(indices_with_space)

            indices = [
                indices[0] for indices in tokenizer.encode_batch(names_with_space)
            ]

            return cls(names_with_space, indices)

    def sample(self, num_names: int) -> NameSample:
        sample_indices = random.sample(range(len(self.indices)), num_names)
        names_with_space = [self.names_with_space[i] for i in sample_indices]
        indices = [self.indices[i] for i in sample_indices]
        return NameSample(names_with_space, indices)

    def sample_batch(self, num_names: int, batch_size: int) -> list[NameSample]:
        batches = []
        for _ in range(batch_size):
            batches.append(self.sample(num_names))

        return batches
