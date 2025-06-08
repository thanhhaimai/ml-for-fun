import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:
    """
    A tokenizer that converts a sentence to a list of token indices and vice versa.

    B: batch size
    S: number of tokens in the sentence, also known as the time dimension T
    V: vocab size
    """

    PAD_TOKEN = "<|pad|>"
    START_TOKEN = "<|start|>"
    END_TOKEN = "<|end|>"

    def __init__(
        self,
        use_start_token: bool = False,
        use_end_token: bool = False,
    ):
        self._special_tokens = []
        self._special_tokens.append(self.PAD_TOKEN)
        if use_start_token:
            self._special_tokens.append(self.START_TOKEN)
        if use_end_token:
            self._special_tokens.append(self.END_TOKEN)

    def load(self, data: list[str]):
        self.index_to_token = self._special_tokens + data
        self.token_to_index: dict[str, int] = {
            c: i for i, c in enumerate(self.index_to_token)
        }
        self.vocab_size = len(self.index_to_token)

        self._pad_token_idx = self.token_to_index[self.PAD_TOKEN]

    @property
    def pad_token_idx(self) -> int:
        return self._pad_token_idx

    def is_special_idx(self, idx: int) -> bool:
        return idx < len(self._special_tokens)

    def t2i(self, s: str) -> list[int]:
        if s in self._special_tokens:
            return [self.token_to_index[s]]

        return [self.token_to_index[c] for c in s]

    def i2t(self, indices: list[int]) -> str:
        return "".join(self.index_to_token[i] for i in indices)

    def to_one_hot(
        self,
        s: str,
        batch_dim: int | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        return:
            shape [S, V] if batch_dim is None
            shape [1, S, V] if batch_dim == 0
            shape [S, 1, V] if batch_dim == 1
        """
        one_hot = F.one_hot(
            torch.tensor(self.t2i(s), device=device),
            num_classes=self.vocab_size,
        ).float()

        if batch_dim is not None:
            return one_hot.unsqueeze(batch_dim)

        return one_hot

    def to_one_hot_batch(
        self,
        sentences: list[str],
        batch_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        return: all sentences padded to the same length
            shape [B, S, V] if batch_dim == 0
            shape [S, B, V] if batch_dim == 1
        """
        if batch_dim == 0:
            batch_first = True
        elif batch_dim == 1:
            batch_first = False
        else:
            raise ValueError(f"Unsupported batch_dim: {batch_dim}")

        one_hots = [self.to_one_hot(sentence, device=device) for sentence in sentences]
        return pad_sequence(
            sequences=one_hots,
            batch_first=batch_first,
            padding_value=self._pad_token_idx,
        )

    def from_one_hot(self, one_hot: torch.Tensor, batch_dim: int | None = None) -> str:
        """
        one_hot:
            shape [S, V] if batch_dim is None
            shape [1, S, V] if batch_dim == 0
            shape [S, 1, V] if batch_dim == 1

        """
        if batch_dim is not None:
            one_hot = one_hot.squeeze(batch_dim)

        padded_token_indices = one_hot.argmax(dim=1)
        token_indices = padded_token_indices[
            padded_token_indices != self._pad_token_idx
        ]
        return self.i2t(token_indices.tolist())

    def from_one_hot_batch(
        self,
        batch_dim: int,
        one_hots: torch.Tensor,
    ) -> list[str]:
        """
        one_hots:
            shape [B, S, V] if batch_dim == 0
            shape [S, B, V] if batch_dim == 1

        return:
            list of strings after removing the padding tokens
        """
        return [
            self.from_one_hot(one_hot, batch_dim)
            for one_hot in one_hots.split(1, dim=batch_dim)
        ]
