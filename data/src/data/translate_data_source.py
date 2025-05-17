from typing import Self


class TranslateDataSource:
    def __init__(
        self,
        source_ids: list[str],
        target_ids: list[str],
        source_sentences: list[str],
        target_sentences: list[str],
    ):
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    @classmethod
    def load(
        cls,
        file_path: str,
    ) -> Self:
        """
        Reads and processes sentence pairs from the specified file.

        File line format:
        <source_id>\t<source_sentence>\t<target_id>\t<target_sentence>\n

        Args:
            file_path: Path to the tab-separated file containing sentence pairs.
        """
        source_ids = []
        target_ids = []
        source_sentences = []
        target_sentences = []

        with open(file_path, "r") as file:
            for line in file:
                source_id, source_sentence, target_id, target_sentence = (
                    line.strip().split("\t")
                )
                source_ids.append(source_id)
                target_ids.append(target_id)
                source_sentences.append(source_sentence)
                target_sentences.append(target_sentence)

        return cls(source_ids, target_ids, source_sentences, target_sentences)
