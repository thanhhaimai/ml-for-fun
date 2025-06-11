import tempfile
from pathlib import Path

import pytest
import tiktoken

from learning.gpt2.popular_names_data_source import PopularNamesDataSource


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def sample_single_token_names():
    # Verified on https://tiktokenizer.vercel.app/?model=gpt2
    return [
        "John",
        "Mary",
        "Alice",
        "Bob",
        "Charlie",
        "Emma",
        "Oliver",
        "Sophia",
        "William",
    ]


@pytest.fixture
def sample_multi_token_names():
    # Verified on https://tiktokenizer.vercel.app/?model=gpt2
    return [
        "Vivian",
        "Jasmine",
        "Isabella",
        "Abigail",
    ]


@pytest.fixture
def sample_file(sample_single_token_names, sample_multi_token_names):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Contains both duplicates and multi-token names
        content = "\n".join(
            sample_single_token_names
            + sample_multi_token_names
            + sample_single_token_names
        )
        f.write(content)
        f.flush()
        yield f.name

    Path(f.name).unlink()


@pytest.fixture
def empty_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        f.flush()
        yield f.name

    Path(f.name).unlink()


def test_init_basic():
    names = [" Alice", " Bob", " Charlie"]
    data_source = PopularNamesDataSource(names)
    assert data_source.names_with_space == names


def test_load_basic_names(sample_file, tokenizer, sample_single_token_names):
    data_source = PopularNamesDataSource.load(sample_file, tokenizer)

    print(data_source.names_with_space)
    print(sample_single_token_names)

    # All names should be prefixed with space and be single tokens
    assert len(data_source.names_with_space) == len(sample_single_token_names)
    assert all(name.startswith(" ") for name in data_source.names_with_space)
    assert all(
        len(tokenizer.encode(name)) == 1 for name in data_source.names_with_space
    )
    assert sorted(data_source.names_with_space) == sorted(
        [f" {name}" for name in sample_single_token_names]
    )


def test_load_empty_file(empty_file, tokenizer):
    data_source = PopularNamesDataSource.load(empty_file, tokenizer)
    assert data_source.names_with_space == []
