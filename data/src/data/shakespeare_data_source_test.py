import pytest

from data.shakespeare_data_source import ShakespeareDataSource
from data.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer()


@pytest.fixture
def shakespeare_file(tmp_path):
    data_dir = tmp_path / "shakespeare_data"
    data_dir.mkdir()
    file_path = data_dir / "sample.txt"
    # A small snippet of Shakespeare-like text
    text_content = "Hello, world!\nAlas, poor Yorick."
    file_path.write_text(text_content)
    return str(file_path), text_content


def test_load_basic(shakespeare_file, tokenizer: Tokenizer):
    file_path, text_content = shakespeare_file
    ds = ShakespeareDataSource.load(file_path, tokenizer)

    assert ds.text == text_content

    # Verify tokenizer is loaded
    expected_vocab = set(text_content)

    # Tokenizer adds PAD_TOKEN by default
    assert tokenizer.PAD_TOKEN in ds.tokenizer.token_to_index
    for char in expected_vocab:
        assert char in ds.tokenizer.token_to_index

    # Verify token_frequency
    # The length of token_frequency should be the vocab size (including special tokens)
    assert len(ds.token_frequency) == ds.tokenizer.vocab_size

    # Check frequencies of actual characters from the text
    for token, index in ds.tokenizer.token_to_index.items():
        if token in ds.token_counter:
            assert ds.token_frequency[index] == ds.token_counter[token]
        elif token == tokenizer.PAD_TOKEN:
            # PAD_TOKEN is not in the text, so its count should be 0 in token_counter
            # but it exists in token_frequency with a count derived from token_counter behavior
            assert ds.token_frequency[index] == ds.token_counter.get(token, 0)
        # Add checks for other special tokens if they were enabled in the tokenizer
