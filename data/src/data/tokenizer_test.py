import pytest
import torch

from data.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    t = Tokenizer()
    data = set()
    data.update("hello")
    data.update("world")
    t.load(data)
    return t


def test_load(tokenizer: Tokenizer):
    assert set(tokenizer.index_to_token) == set("helowrd") | {Tokenizer.PAD_TOKEN}
    assert len(tokenizer.index_to_token) == len(set("helowrd")) + 1
    assert tokenizer.vocab_size == len(tokenizer.token_to_index)
    for char, index in tokenizer.token_to_index.items():
        assert tokenizer.index_to_token[index] == char


def test_load_start_end():
    tokenizer = Tokenizer(use_start_token=True, use_end_token=True)
    data = set()
    data.update("hello")
    data.update("world")
    tokenizer.load(data)
    assert set(tokenizer.index_to_token) == set("helowrd") | {
        Tokenizer.PAD_TOKEN,
        Tokenizer.START_TOKEN,
        Tokenizer.END_TOKEN,
    }
    assert len(tokenizer.index_to_token) == len(set("helowrd")) + 3
    assert tokenizer.vocab_size == len(tokenizer.token_to_index)
    for char, index in tokenizer.token_to_index.items():
        assert tokenizer.index_to_token[index] == char


def test_t2i(tokenizer: Tokenizer):
    indices = tokenizer.t2i("hello")
    assert isinstance(indices, list)
    assert len(indices) == 5
    for index in indices:
        assert isinstance(index, int)
        assert 0 <= index < tokenizer.vocab_size


def test_i2t(tokenizer: Tokenizer):
    original_string = "world"
    indices = tokenizer.t2i(original_string)
    reconstructed_string = tokenizer.i2t(indices)
    assert reconstructed_string == original_string

    assert (
        tokenizer.i2t([tokenizer.token_to_index[Tokenizer.PAD_TOKEN]])
        == Tokenizer.PAD_TOKEN
    )


def test_t2i_empty_string(tokenizer: Tokenizer):
    assert tokenizer.t2i("") == []


def test_i2t_empty_list(tokenizer: Tokenizer):
    assert tokenizer.i2t([]) == ""


def test_to_one_hot(tokenizer: Tokenizer):
    one_hot = tokenizer.to_one_hot("hello")
    assert isinstance(one_hot, torch.Tensor)
    assert one_hot.shape == (5, tokenizer.vocab_size)
    assert one_hot.dtype == torch.float32

    # Check that each row is a valid one-hot encoding
    for i in range(5):
        assert one_hot[i].sum() == 1.0
        assert torch.all((one_hot[i] == 0) | (one_hot[i] == 1))


def test_to_one_hot_batch(tokenizer: Tokenizer):
    sentences = ["hello", "world"]
    batch_one_hot = tokenizer.to_one_hot_batch(sentences, batch_dim=0)
    assert isinstance(batch_one_hot, torch.Tensor)
    assert batch_one_hot.shape == (2, 5, tokenizer.vocab_size)
    assert batch_one_hot.dtype == torch.float32

    # Verify each sentence in the batch
    for i, sentence in enumerate(sentences):
        expected_one_hot = tokenizer.to_one_hot(sentence)
        assert torch.equal(batch_one_hot[i], expected_one_hot)


def test_to_one_hot_batch_different_lengths():
    t = Tokenizer()
    data = set()
    data.update("abc")
    data.update("de")
    data.update("fghi")
    t.load(data)
    sentences = ["abc", "de", "f", "abcdghi"]
    one_hots = t.to_one_hot_batch(sentences, batch_dim=0)
    assert one_hots.shape == (4, 7, t.vocab_size)
    assert one_hots.dtype == torch.float32


def test_from_one_hot(tokenizer: Tokenizer):
    original_string = "world"
    one_hot = tokenizer.to_one_hot(original_string)
    reconstructed_string = tokenizer.from_one_hot(one_hot)
    assert reconstructed_string == original_string


def test_from_one_hot_batch(tokenizer: Tokenizer):
    sentences = ["hello", "world", "helloworld"]
    one_hots = tokenizer.to_one_hot_batch(sentences, batch_dim=0)
    reconstructed_sentences = tokenizer.from_one_hot_batch(
        batch_dim=0, one_hots=one_hots
    )
    assert reconstructed_sentences == sentences
