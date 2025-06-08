from collections import Counter
from typing import Self


class ShakespeareDataSource:
    def __init__(
        self,
        text: str,
        vocab: list[str],
    ):
        self.text = text
        self.vocab = vocab
        self.token_counter = Counter(text)

    @classmethod
    def load(
        cls,
        file_path: str,
    ) -> Self:
        """
        Loads the Shakespeare text from the specified file and creates a ShakespeareDataSource object.

        Args:
            file_path: The path to the file containing the Shakespeare text.
            tokenizer: Simple character tokenizer.
        """
        with open(file_path, "r") as file:
            text = file.read()

        vocab = sorted(set(text))
        return cls(text, vocab)
