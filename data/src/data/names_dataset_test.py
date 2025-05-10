import os

import pytest
import torch

from data.names_data_source import NamesDataSource, unicode_to_ascii
from data.names_dataset import NameSample, NamesDataset


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
    return os.path.join(os.path.dirname(__file__), "../../../datasets/names")


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
    ds = NamesDataSource.load(str(test_dir))
    dataset = NamesDataset(ds)
    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample, NameSample)
    assert sample.name in ["John", "Jane", "Jean", "Marie"]
    assert sample.country in ["English", "French"]
    assert sample.country_tensor.shape == (1,)
    assert sample.name_tensor.shape[1] == 1
    assert sample.name_tensor.shape[2] == ds.num_vocab


def test_names_dataset_name_to_one_hot_and_back(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    dataset = NamesDataset(ds)
    name = "John"
    one_hot = dataset.name_to_one_hot(name)
    assert one_hot.shape[0] == len(name)
    assert one_hot.shape[2] == ds.num_vocab
    # Remove batch dim for one_hot_to_name
    recovered = dataset.one_hot_to_name(one_hot.squeeze(1))
    assert recovered == name


def test_names_dataset_country_index_to_one_hot(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    dataset = NamesDataset(ds)
    for idx in range(ds.num_classes):
        one_hot = dataset.country_index_to_one_hot(idx)
        assert one_hot.shape[0] == ds.num_classes
        assert one_hot[idx] == 1.0
        assert one_hot.sum() == 1.0


def test_names_dataset_out_of_range(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    dataset = NamesDataset(ds)
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
    dataset = NamesDataset(ds)
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, NameSample)
    assert isinstance(sample.name, str)
    assert isinstance(sample.country, str)
    assert isinstance(sample.country_tensor, torch.Tensor)
    assert isinstance(sample.name_tensor, torch.Tensor)


def test_names_data_source_empty_folder(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = NamesDataSource.load(str(empty_dir))
    assert ds.num_classes == 0
    assert ds.num_vocab == 0
    assert ds.countries == []
    assert ds.country_idx_to_names == {}
