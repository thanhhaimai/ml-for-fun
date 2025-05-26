import os

import pytest

from data.shakespeare_data_source import ShakespeareDataSource
from data.tokenizer import Tokenizer
from learning.shakespeare_generator.shakespeare_dataset import (
    Sample,
    ShakespeareDataset,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../../../datasets")


@pytest.fixture
def tokenizer() -> Tokenizer:
    # Using default tokenizer settings for Shakespeare (no start/end tokens usually)
    return Tokenizer()


@pytest.fixture
def shakespeare_file_and_content(tmp_path):
    data_dir = tmp_path / "shakespeare_data"
    data_dir.mkdir()
    file_path = data_dir / "sample.txt"
    text_content = "Hello, world!"  # Length 13
    file_path.write_text(text_content)
    return str(file_path), text_content


@pytest.fixture
def shakespeare_data_source(
    shakespeare_file_and_content, tokenizer: Tokenizer
) -> ShakespeareDataSource:
    file_path, _ = shakespeare_file_and_content
    return ShakespeareDataSource.load(file_path, tokenizer)


def test_basic_dataset_creation(
    shakespeare_data_source: ShakespeareDataSource,
    tokenizer: Tokenizer,
    shakespeare_file_and_content,
):
    _, text_content = shakespeare_file_and_content
    sequence_length = 5
    dataset = ShakespeareDataset(shakespeare_data_source, tokenizer, sequence_length)

    # Expected number of samples: len(text) - sequence_length
    assert len(dataset) == len(text_content) - sequence_length  # 13 - 5 = 8

    sample = dataset[0]
    assert isinstance(sample, Sample)
    assert sample.input.shape == (sequence_length, tokenizer.vocab_size)
    assert sample.label.shape == (sequence_length,)

    # Check content of the first sample
    # Input: "Hello"
    # Label: "ello,"
    expected_input_text = text_content[:sequence_length]  # "Hello"
    expected_label_text = text_content[1 : sequence_length + 1]  # "ello,"

    assert tokenizer.from_one_hot(sample.input) == expected_input_text
    assert tokenizer.i2t(sample.label.tolist()) == expected_label_text

    # Check content of the last sample
    # text_content = "Hello, world!" (len 13)
    # sequence_length = 5
    # last sample index = 13 - 5 - 1 = 7
    # input: text_content[7:12] = "world"
    # label: text_content[8:13] = "orld!"
    last_sample = dataset[len(text_content) - sequence_length - 1]
    expected_last_input_text = text_content[
        len(text_content) - sequence_length - 1 : len(text_content) - 1
    ]
    expected_last_label_text = text_content[
        len(text_content) - sequence_length : len(text_content)
    ]

    assert tokenizer.from_one_hot(last_sample.input) == expected_last_input_text
    assert tokenizer.i2t(last_sample.label.tolist()) == expected_last_label_text


def test_out_of_range(
    shakespeare_data_source: ShakespeareDataSource, tokenizer: Tokenizer
):
    sequence_length = 5
    dataset = ShakespeareDataset(shakespeare_data_source, tokenizer, sequence_length)
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]  # Try to access one element beyond the end


def test_empty_file(tmp_path, tokenizer: Tokenizer):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    empty_ds = ShakespeareDataSource.load(str(empty_file), tokenizer)
    # Tokenizer will have PAD_TOKEN
    assert tokenizer.vocab_size > 0

    sequence_length = 5
    dataset = ShakespeareDataset(empty_ds, tokenizer, sequence_length)
    assert len(dataset) == 0  # No samples can be created from empty text

    with pytest.raises(IndexError):
        _ = dataset[0]


def test_sequence_length_equals_text_length(
    shakespeare_data_source: ShakespeareDataSource,
    tokenizer: Tokenizer,
    shakespeare_file_and_content,
):
    _, text_content = shakespeare_file_and_content
    sequence_length = len(text_content)  # 13
    dataset = ShakespeareDataset(shakespeare_data_source, tokenizer, sequence_length)

    # len(text_content) - sequence_length = 0
    assert len(dataset) == 0

    with pytest.raises(IndexError):
        _ = dataset[0]


def test_sequence_length_greater_than_text_length(
    shakespeare_data_source: ShakespeareDataSource,
    tokenizer: Tokenizer,
    shakespeare_file_and_content,
):
    _, text_content = shakespeare_file_and_content
    sequence_length = len(text_content) + 1  # 14
    dataset = ShakespeareDataset(shakespeare_data_source, tokenizer, sequence_length)

    # len(text_content) - sequence_length should be < 0, resulting in 0 samples
    assert len(dataset) == 0

    with pytest.raises(IndexError):
        _ = dataset[0]
