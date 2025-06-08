import os

import pytest

from data.names_data_source import NamesDataSource, unicode_to_ascii


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
    assert ds.country_idx_to_names[0] == ["Jane", "John"]
    assert ds.country_idx_to_names[1] == ["Jean", "Marie"]
    assert set("JohnJaneJeanMarie").issubset(ds.vocab)
    assert ds.num_classes == 2


def test_unicode_to_ascii():
    assert unicode_to_ascii("éèêëç") == "eeeec"
    assert unicode_to_ascii("Đặng") == "Đang"
    assert unicode_to_ascii("François") == "Francois"


def test_normalize_unicode(test_dir):
    (test_dir / "Vietnamese.txt").write_text("Đặng\n")
    ds = NamesDataSource.load(str(test_dir), normalize_unicode=True)
    assert "Đang" in ds.country_idx_to_names[2]
    assert "Đ" in ds.vocab
    assert "ạ" not in ds.vocab
    assert "ă" not in ds.vocab
    assert "g" in ds.vocab


def test_empty_folder(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = NamesDataSource.load(str(empty_dir))
    assert ds.num_classes == 0
    assert ds.countries == []
    assert ds.country_idx_to_names == {}


def test_real_data(real_data_dir):
    ds = NamesDataSource.load(real_data_dir)
    assert ds.num_classes > 0
    assert len
    assert all(isinstance(c, str) for c in ds.countries)
    assert all(isinstance(names, list) for names in ds.country_idx_to_names.values())
    for names in ds.country_idx_to_names.values():
        for name in names:
            assert isinstance(name, str)
            assert name
