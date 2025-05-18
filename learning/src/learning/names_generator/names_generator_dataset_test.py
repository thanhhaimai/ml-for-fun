import os

import pytest
import torch

from data.names_data_source import (
    NamesDataSource,
)
from data.tokenizer import Tokenizer
from learning.names_generator.names_generator_dataset import (
    NameSample,
    NamesGeneratorDataset,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../../../datasets")


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(use_start_token=True, use_end_token=True)


@pytest.fixture
def test_dir(tmp_path):
    test_dir = tmp_path / "test_names_data"
    test_dir.mkdir()
    (test_dir / "English.txt").write_text("John\nJane\n")
    (test_dir / "French.txt").write_text("Jean\nMarie\n")
    return test_dir


@pytest.fixture
def real_data_dir():
    return os.path.join(DATA_ROOT, "names")


def test_basic(test_dir, tokenizer: Tokenizer):
    ds = NamesDataSource.load(str(test_dir), tokenizer=tokenizer)
    dataset = NamesGeneratorDataset(ds, tokenizer=tokenizer)
    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample, NameSample)

    # First sample is English, Jane
    # Input: <|start|>, J,a,n,e (len 5), Label: J,a,n,e,<|end|> (len 5)
    # Vocab: <|start|>, <|end|>, J,o,h,n,a,e,M,r,i, (11 tokens)
    print(sample)
    assert sample.category.shape == (1, 2)
    assert sample.input.shape == (5, tokenizer.vocab_size)
    assert sample.label.shape == (5,)

    assert tokenizer.from_one_hot(sample.input) == "<|start|>Jane"
    assert tokenizer.i2t(sample.label.tolist()) == "Jane<|end|>"


def test_out_of_range(test_dir, tokenizer: Tokenizer):
    ds = NamesDataSource.load(str(test_dir), tokenizer=tokenizer)
    dataset = NamesGeneratorDataset(ds, tokenizer=tokenizer)
    with pytest.raises(IndexError):
        _ = dataset[100]


def test_real_data(real_data_dir, tokenizer: Tokenizer):
    ds = NamesDataSource.load(real_data_dir, tokenizer=tokenizer)
    dataset = NamesGeneratorDataset(ds, tokenizer=tokenizer)
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, NameSample)
    assert isinstance(sample.label, torch.Tensor)
    assert isinstance(sample.input, torch.Tensor)


def test_empty_folder(tmp_path, tokenizer: Tokenizer):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = NamesDataSource.load(str(empty_dir), tokenizer=tokenizer)
    assert ds.num_classes == 0
    assert ds.countries == []
    assert ds.country_idx_to_names == {}
