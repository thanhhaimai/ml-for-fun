import tempfile
from pathlib import Path

import pytest
import tiktoken

from learning.ioi_circuit.data_sources import NamesDataSource


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


def test_load_basic_names(sample_file, tokenizer, sample_single_token_names):
    data_source = NamesDataSource.load(sample_file, tokenizer)

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

    # The indices should be the indices of the names in the tokenizer
    assert data_source.indices == [
        tokenizer.encode(name)[0] for name in data_source.names_with_space
    ]


def test_load_empty_file(empty_file, tokenizer):
    data_source = NamesDataSource.load(empty_file, tokenizer)
    assert data_source.names_with_space == []


def test_sample(sample_file, tokenizer):
    data_source = NamesDataSource.load(sample_file, tokenizer)
    sample = data_source.sample(3)
    assert len(sample.names_with_space) == 3
    assert len(sample.indices) == 3
    for name, index in zip(sample.names_with_space, sample.indices):
        assert tokenizer.encode(name)[0] == index


def test_sample_batch(sample_file, tokenizer):
    data_source = NamesDataSource.load(sample_file, tokenizer)
    batch = data_source.sample_batch(3, 2)
    assert len(batch) == 2
    assert all(len(sample.names_with_space) == 3 for sample in batch)
    assert all(len(sample.indices) == 3 for sample in batch)

    for sample in batch:
        for name, index in zip(sample.names_with_space, sample.indices):
            assert tokenizer.encode(name)[0] == index
