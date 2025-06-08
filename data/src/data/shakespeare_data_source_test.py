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
    ds = ShakespeareDataSource.load(file_path)
    vocab = sorted(set(text_content))
    tokenizer.load(vocab)

    assert ds.text == text_content

    # Verify tokenizer is loaded
    expected_vocab = set(text_content)

    # Tokenizer adds PAD_TOKEN by default
    assert tokenizer.PAD_TOKEN in tokenizer.token_to_index
    for char in expected_vocab:
        assert char in tokenizer.token_to_index
