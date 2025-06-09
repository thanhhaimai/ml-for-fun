import pytest
import tiktoken
import torch

from learning.gpt2.ioi_circuit_analyzer import (
    IoiCircuitAnalyzer,
    NameSampler,
    PromptTemplate,
)
from learning.gpt2.model import GPT2, PretrainedName


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
def sampler() -> NameSampler:
    return NameSampler(names=["Mary", "John", "Tom", "Jerry"])


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_topk_logits(
    model: GPT2,
    tokenizer: tiktoken.Encoding,
    sampler: NameSampler,
    device: torch.device,
):
    prompt_template = PromptTemplate(
        template="When {s1} and {s2} went to the store, {s3} gave a drink to",
        name_sampler=sampler,
    )
    analyzer = IoiCircuitAnalyzer(model, tokenizer, prompt_template, device)
    result = analyzer.topk_logits(prompt_template.from_abb("Mary", "John"), k=1)

    assert len(result.top_probs) == 1
    assert len(result.top_indices) == 1

    assert result.top_probs[0] > 0.0
    decoded_token = tokenizer.decode(result.top_indices.tolist())
    # There is a space in front of the token because of the GPT2 tokenizer
    assert decoded_token == " Mary"
