import pytest
import tiktoken
import torch

from learning.gpt2.data_sources import NamesDataSource
from learning.gpt2.ioi_circuit_analyzer import IoiCircuitAnalyzer
from learning.gpt2.model import GPT2, PretrainedName
from learning.gpt2.prompts import PromptTemplate


@pytest.fixture
def model() -> GPT2:
    model, _pretrained_model = GPT2.from_pretrained(
        PretrainedName.GPT2_SMALL,
        device=torch.device("cpu"),
    )
    return model


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def data_source(tokenizer: tiktoken.Encoding) -> NamesDataSource:
    names = ["Mary", "John", "Tom", "Jerry"]
    names_with_space = [f" {name}" for name in names]
    indices = [indices[0] for indices in tokenizer.encode_batch(names_with_space)]
    return NamesDataSource(names_with_space, indices)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_topk_logits(
    model: GPT2,
    tokenizer: tiktoken.Encoding,
    data_source: NamesDataSource,
    device: torch.device,
):
    prompt_template = PromptTemplate(
        template="When {s1} and {s2} went to the store, {s3} gave a drink to",
        names_data_source=data_source,
        device=device,
    )
    analyzer = IoiCircuitAnalyzer(model, tokenizer, prompt_template, device)
    result = analyzer.topk_probs(prompt_template.from_abb("Mary", "John"), k=1)

    assert len(result.top_probs) == 1
    assert len(result.top_indices) == 1

    assert result.top_probs[0] > 0.0
    decoded_token = tokenizer.decode(result.top_indices.tolist())
    # There is a space in front of the token because of the GPT2 tokenizer
    assert decoded_token == " Mary"
