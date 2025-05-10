import os

import pytest

from data.names_data_source import NamesDataSource, unicode_to_ascii


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


def test_load_basic(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    assert ds.countries == ["English", "French"]
    assert ds.country_idx_to_names[0] == ["John", "Jane"]
    assert ds.country_idx_to_names[1] == ["Jean", "Marie"]
    assert set(ds.tokens) >= set("JohnJaneJeanMarie")
    assert ds.num_classes == 2
    assert ds.num_vocab == len(ds.tokens)


def test_unicode_to_ascii():
    assert unicode_to_ascii("éèêëç") == "eeeec"
    assert unicode_to_ascii("Đặng") == "Đang"
    assert unicode_to_ascii("François") == "Francois"


def test_prefix_suffix(test_dir):
    ds = NamesDataSource.load(str(test_dir), prefix="*", suffix="$")
    for names in ds.country_idx_to_names.values():
        for name in names:
            assert name.startswith("*")
            assert name.endswith("$")


def test_normalize_unicode(test_dir):
    # Write a name with accents
    (test_dir / "Vietnamese.txt").write_text("Đặng\n")
    ds = NamesDataSource.load(str(test_dir), normalize_unicode=True)
    found = any("Đang" in name for name in ds.country_idx_to_names[2])
    assert found


def test_tokenization_and_detokenization(test_dir):
    ds = NamesDataSource.load(str(test_dir))
    name = "John"
    indices = ds.t2i(name)
    assert isinstance(indices, list)
    assert ds.i2t(indices) == name


def test_empty_folder(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = NamesDataSource.load(str(empty_dir))
    assert ds.num_classes == 0
    assert ds.num_vocab == 0
    assert ds.countries == []
    assert ds.country_idx_to_names == {}


def test_real_data(real_data_dir):
    ds = NamesDataSource.load(real_data_dir)
    assert ds.num_classes > 0
    assert ds.num_vocab > 0
    assert all(isinstance(c, str) for c in ds.countries)
    assert all(isinstance(names, list) for names in ds.country_idx_to_names.values())
    # Check that all names are strings and non-empty
    for names in ds.country_idx_to_names.values():
        for name in names:
            assert isinstance(name, str)
            assert name
