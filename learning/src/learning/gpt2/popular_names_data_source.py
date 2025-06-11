from typing import Self

import tiktoken


class PopularNamesDataSource:
    """
    Data source for popular names that ensures all names (with space) are single tokens.

    All names are stored with a space prefix.
    """

    def __init__(
        self,
        names_with_space: list[str],
    ):
        self.names_with_space = names_with_space

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

            return cls(names_with_space)
