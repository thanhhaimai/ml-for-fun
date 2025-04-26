import os

import pytest
import torch

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
    return os.path.join(os.path.dirname(__file__), "../../../datasets/names")


def test_dataset_initialization(test_dir):
    dataset = NamesDataset(data_folder=test_dir)
    assert len(dataset) == 4  # 2 names per country, 2 countries
    assert dataset.countries == ["English", "French"]
    assert dataset.names == [
        ("John", 0),
        ("Jane", 0),
        ("Jean", 1),
        ("Marie", 1),
    ]


def test_get_item_max_countries_count(test_dir):
    dataset = NamesDataset(data_folder=test_dir, max_countries_count=1)

    name_tensor, country_tensor = dataset[0]
    country_index = int(country_tensor.squeeze(0).item())
    assert dataset.tensor_to_name(name_tensor) == dataset.names[0][0]
    assert country_index == dataset.names[0][1]

    name_tensor, country_tensor = dataset[1]
    country_index = int(country_tensor.squeeze(0).item())
    assert dataset.tensor_to_name(name_tensor) == dataset.names[1][0]
    assert country_index == dataset.names[1][1]

    with pytest.raises(IndexError):
        dataset[2]


def test_get_item_max_names_count(test_dir):
    dataset = NamesDataset(data_folder=test_dir, max_names_count=1)

    name_tensor, country_tensor = dataset[0]
    country_index = int(country_tensor.squeeze(0).item())
    assert dataset.tensor_to_name(name_tensor) == "John"
    assert dataset.countries[country_index] == "English"

    with pytest.raises(IndexError):
        dataset[1]


def test_index_out_of_range(test_dir):
    dataset = NamesDataset(data_folder=test_dir)
    with pytest.raises(IndexError):
        dataset[10]


def test_non_existent_folder():
    with pytest.raises(FileNotFoundError):
        NamesDataset(data_folder="/non/existent/folder")


def test_dataset_with_transform(test_dir):
    dataset = NamesDataset(
        data_folder=test_dir,
        transform_input=lambda x: x.upper(),
    )
    assert dataset.names == [
        ("JOHN", 0),
        ("JANE", 0),
        ("JEAN", 1),
        ("MARIE", 1),
    ]


def test_real_data_initialization(real_data_dir):
    dataset = NamesDataset(
        data_folder=real_data_dir, max_countries_count=5, max_names_count=10
    )
    assert len(dataset) == 10  # We have enough real data for 10 names
    assert len(dataset.countries) == 1  # The first country already exhausts 10 names


def test_real_data_get_item(real_data_dir):
    dataset = NamesDataset(
        data_folder=real_data_dir,
        max_countries_count=5,
        max_names_count=10,
    )
    name_tensor, country_tensor = dataset[0]
    country_index = int(country_tensor.squeeze(0).item())
    name = dataset.tensor_to_name(name_tensor)
    assert (name, country_index) in dataset.names


def test_name_to_tensor(test_dir):
    dataset = NamesDataset(data_folder=test_dir)
    tensor = dataset.name_to_tensor("John")
    assert tensor.shape[0] == len("John")  # Check sequence length dimension
    assert tensor.shape[1] == 1  # Check batch size dimension
    assert tensor.shape[2] == len(dataset.index_to_token)  # Check token dimension


def test_tensor_to_name(test_dir):
    dataset = NamesDataset(data_folder=test_dir)
    tensor = dataset.name_to_tensor("John")
    name = dataset.tensor_to_name(tensor)
    assert name == "John"


def test_country_index_to_tensor(test_dir):
    dataset = NamesDataset(data_folder=test_dir)
    tensor = dataset.country_index_to_tensor(0)
    assert tensor == torch.tensor([[0]])


def test_to_method(test_dir):
    dataset = NamesDataset(data_folder=test_dir)
    device = torch.device("cpu")

    # Move tensors to CPU
    dataset.to(device)
    assert dataset.names_tensors[0].device == device
    assert dataset.countries_tensors[0].device == device

    # Move tensors again to the same device
    dataset.to(device)  # Should not perform unnecessary operations
    assert dataset.names_tensors[0].device == device
    assert dataset.countries_tensors[0].device == device
