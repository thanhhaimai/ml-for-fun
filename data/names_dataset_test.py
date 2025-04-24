import os

import pytest

from data.names_dataset import NamesDataset


@pytest.fixture
def test_dir(tmp_path):
    # Create a temporary directory with sample data files
    test_dir = tmp_path / "test_names_data"
    test_dir.mkdir()

    # Create sample country files
    # English will be processed first because we process files in sorted order
    (test_dir / "English.txt").write_text("John\nJane\n")
    (test_dir / "French.txt").write_text("Jean\nMarie\n")

    return test_dir


@pytest.fixture
def real_data_dir():
    # Path to the real data directory
    return os.path.join(os.path.dirname(__file__), "../datasets/names")


def test_dataset_initialization(test_dir):
    dataset = NamesDataset(
        data_folder=test_dir, max_countries_count=10, max_names_count=10
    )
    assert len(dataset) == 4  # 2 names per country, 2 countries
    assert dataset.countries == ["English", "French"]
    assert dataset.names == [
        ("John", 0),
        ("Jane", 0),
        ("Jean", 1),
        ("Marie", 1),
    ]


def test_get_item(test_dir):
    dataset = NamesDataset(
        data_folder=test_dir, max_countries_count=2, max_names_count=2
    )
    name, country = dataset[0]
    assert name == "John"
    assert country == "English"

    name, country = dataset[1]
    assert name == "Jane"
    assert country == "English"

    with pytest.raises(IndexError):
        dataset[2]


def test_index_out_of_range(test_dir):
    dataset = NamesDataset(
        data_folder=test_dir, max_countries_count=2, max_names_count=10
    )
    with pytest.raises(IndexError):
        dataset[10]


def test_real_data_initialization(real_data_dir):
    dataset = NamesDataset(
        data_folder=real_data_dir, max_countries_count=5, max_names_count=10
    )
    print(dataset.names)
    assert len(dataset) == 10  # We have enough real data for 50 names
    assert len(dataset.countries) == 1  # The first country already exhausts 10 names


def test_real_data_get_item(real_data_dir):
    dataset = NamesDataset(
        data_folder=real_data_dir, max_countries_count=5, max_names_count=10
    )
    name, country = dataset[0]
    assert isinstance(name, str)  # Ensure name is a string
    assert isinstance(country, str)  # Ensure country is a string
    assert country in dataset.countries  # Ensure country is valid


def test_real_data_max_names_per_country(real_data_dir):
    dataset = NamesDataset(
        data_folder=real_data_dir, max_countries_count=5, max_names_count=3
    )
    for country in dataset.countries:
        country_names = [name for name, c in dataset if c == country]
        assert len(country_names) <= 3  # Ensure max_names_count is respected
