import pytest
import tiktoken
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from learning.gpt2.model import GPT2, PretrainedName


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@torch.no_grad()
def test_model_load(tokenizer: tiktoken.Encoding, device: torch.device):
    model, pretrained_model = GPT2.from_pretrained(
        PretrainedName.GPT2_SMALL,
        device=torch.device("cpu"),
    )

    model.eval()
    pretrained_model.eval()

    print("-" * 80)
    print(model)
    print("-" * 80)
    print(pretrained_model)

    names = [
        "Mary",
        "John",
        "Mike",
        "Tom",
        "Jerry",
    ]

    B = len(names)
    V = model.vocab_size

    # shape: [B, S], with S = 1
    data = [tokenizer.encode(name) for name in names]
    print(data)
    indices = torch.tensor(data)
    assert_shape("indices", indices, (B, 1))

    # shape: [B, S, V], with S = 1 (the next token)
    logits = model(indices)
    assert_shape("logits", logits, (B, 1, V))

    probs = F.softmax(logits, dim=-1)

    pretrained_result: CausalLMOutputWithCrossAttentions = pretrained_model(indices)
    pretrained_logits = pretrained_result.logits
    assert pretrained_logits is not None
    assert_shape("pretrained_logits", pretrained_logits, (B, 1, V))

    pretrained_probs = F.softmax(pretrained_logits, dim=-1)

    print("-" * 80)
    print(probs)
    print("-" * 80)
    print(pretrained_probs)

    # Assert that the loaded model's logits are the same as the pretrained model's logits
    print("-" * 80)
    print(logits)
    print("-" * 80)
    print(pretrained_logits)
    assert torch.allclose(logits, pretrained_logits)
