import os

import pytest
import torch

from data.names_data_source import (
    END_TOKEN,
    START_TOKEN,
    NamesDataSource,
    unicode_to_ascii,
)
from learning.names_generator.names_generator_dataset import (
    NameSample,
    NamesGeneratorDataset,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../../../datasets")


def make_data_source(test_dir, **kwargs):
    return NamesDataSource.load(str(test_dir), **kwargs)


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


def test_names_data_source_load(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    assert ds.countries == ["English", "French"]
    assert ds.country_idx_to_names[0] == ["John", "Jane"]
    assert ds.country_idx_to_names[1] == ["Jean", "Marie"]
    assert set(ds.tokens) >= set("JohnJaneJeanMarie")
    assert ds.num_classes == 2
    assert ds.num_vocab == len(ds.tokens)


def test_names_data_source_unicode_ascii():
    assert unicode_to_ascii("éèêëç") == "eeeec"
    assert unicode_to_ascii("Đặng") == "Đang"


def test_names_data_source_prefix_suffix(test_dir):
    ds = NamesDataSource.load(str(test_dir), prefix="*", suffix="$")
    assert all(
        name.startswith("*") and name.endswith("$")
        for names in ds.country_idx_to_names.values()
        for name in names
    )


def test_names_data_source_t2i_i2t(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    name = "John"
    indices = ds.t2i(name)
    assert isinstance(indices, list)
    assert ds.i2t(indices) == name


def test_names_dataset_basic(test_dir):
    ds = NamesDataSource.load(str(test_dir), prefix=START_TOKEN, suffix=END_TOKEN)
    dataset = NamesGeneratorDataset(ds)
    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample, NameSample)

    # First sample is English, .John~
    # The sequence length is 5 because we count the start token for input, and the end token for label
    # There are 9 tokens, and 2 extra tokens for the start and end tokens
    assert sample.category.shape == (5, 2)
    assert sample.input.shape == (5, 11)
    assert sample.label.shape == (5,)


def test_names_dataset_out_of_range(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    dataset = NamesGeneratorDataset(ds)
    with pytest.raises(IndexError):
        _ = dataset[100]


def test_names_data_source_real_data(real_data_dir):
    ds = NamesDataSource.load(real_data_dir)
    assert ds.num_classes > 0
    assert ds.num_vocab > 0
    assert all(isinstance(c, str) for c in ds.countries)
    assert all(isinstance(names, list) for names in ds.country_idx_to_names.values())


def test_names_dataset_real_data(real_data_dir):
    ds = NamesDataSource.load(real_data_dir)
    dataset = NamesGeneratorDataset(ds)
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, NameSample)
    assert isinstance(sample.label, torch.Tensor)
    assert isinstance(sample.input, torch.Tensor)


def test_names_data_source_empty_folder(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = NamesDataSource.load(str(empty_dir))
    assert ds.num_classes == 0
    assert ds.num_vocab == 0
    assert ds.countries == []
    assert ds.country_idx_to_names == {}
