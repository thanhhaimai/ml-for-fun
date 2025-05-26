from collections import Counter
from typing import Self

from data.tokenizer import Tokenizer


class ShakespeareDataSource:
    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
    ):
        self.text = text
        self.tokenizer = tokenizer

        self.token_counter = Counter(text)
        self.token_frequency = [
            self.token_counter[token] for token in self.tokenizer.index_to_token
        ]

    @classmethod
    def load(
        cls,
        file_path: str,
        tokenizer: Tokenizer,
    ) -> Self:
        """
        Loads the Shakespeare text from the specified file and creates a ShakespeareDataSource object.

        Args:
            file_path: The path to the file containing the Shakespeare text.
            tokenizer: Simple character tokenizer.
        """
        with open(file_path, "r") as file:
            text = file.read()

        vocab = set(text)
        tokenizer.load(vocab)
        return cls(text, tokenizer)
