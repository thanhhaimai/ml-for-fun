import pytest
import tiktoken
import torch

from learning.gpt2.ioi_circuit_analyzer import IoiCircuitAnalyzer
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
def device() -> torch.device:
    return torch.device("cpu")


def test_topk_logits(model: GPT2, tokenizer: tiktoken.Encoding, device: torch.device):
    analyzer = IoiCircuitAnalyzer(model, tokenizer, device)
    result = analyzer.topk_logits(
        prompt="When Mary and John went to the store, John gave a drink to",
        k=1,
    )

    assert len(result.top_probs) == 1
    assert len(result.top_indices) == 1

    assert result.top_probs[0] > 0.0
    decoded_token = tokenizer.decode(result.top_indices.tolist())
    # There is a space in front of the token because of the GPT2 tokenizer
    assert decoded_token == " Mary"
