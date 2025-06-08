import pytest
import tiktoken
import torch
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
def test_all_layers_output(tokenizer: tiktoken.Encoding, device: torch.device):
    model, pretrained_model = GPT2.from_pretrained(
        PretrainedName.GPT2_SMALL,
        device=torch.device("cpu"),
    )

    model.eval()
    pretrained_model.eval()

    text = "Mary"
    B = 1
    S = 1
    E = model.embedding_size

    # Due to us not using FlashAttention, the outputs are not exactly the same.
    atol = 1e-5
    rtol = 1e-3

    # ==== Verify Embedding ====
    print("Validating embedding")

    indices = torch.tensor([tokenizer.encode(text)])
    assert_shape("indices", indices, (B, S))

    tokens_embedding = model.embedding(indices)
    assert_shape("tokens_embedding", tokens_embedding, (B, S, E))

    positional_indices = model.positional_indices[:S].expand(B, -1)
    positional_embedding = model.positional_embedding(positional_indices)
    assert_shape("positional_embedding", positional_embedding, (B, S, E))

    embedding = tokens_embedding + positional_embedding
    assert_shape("embedding", embedding, (B, S, E))

    hf_tokens_embedding = pretrained_model.transformer.wte(indices)
    assert_shape("hf_tokens_embedding", hf_tokens_embedding, (B, S, E))

    hf_positional_embedding = pretrained_model.transformer.wpe(
        torch.arange(1).unsqueeze(0)
    )
    assert_shape("hf_positional_embedding", hf_positional_embedding, (B, S, E))

    hf_embedding = hf_tokens_embedding + hf_positional_embedding
    assert_shape("hf_embedding", hf_embedding, (B, S, E))

    assert torch.allclose(embedding, hf_embedding, atol=atol, rtol=rtol)

    # ==== Verify Blocks ====

    output = embedding
    hf_output = hf_embedding
    for i in range(model.config.num_blocks):
        print(f"Validating block {i}")
        output = model.blocks_module[i](output)
        hf_output = pretrained_model.transformer.h[i](hf_output)[0]
        assert torch.allclose(output, hf_output, atol=atol, rtol=rtol)

    # ==== Verify LayerNorm ====

    print("Validating layer norm")
    output = model.layer_norm(output)
    hf_output = pretrained_model.transformer.ln_f(hf_output)
    assert torch.allclose(output, hf_output, atol=atol, rtol=rtol)

    # ==== Verify Logits ====

    print("Validating logits")
    logits = model.linear(output)
    hf_logits = pretrained_model.lm_head(hf_output)
    assert torch.allclose(logits, hf_logits, atol=atol, rtol=rtol)


@torch.no_grad()
def test_same_pretrained_logits(tokenizer: tiktoken.Encoding, device: torch.device):
    model, pretrained_model = GPT2.from_pretrained(
        PretrainedName.GPT2_SMALL,
        device=torch.device("cpu"),
    )

    model.eval()
    pretrained_model.eval()

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

    pretrained_result: CausalLMOutputWithCrossAttentions = pretrained_model(indices)
    pretrained_logits = pretrained_result.logits
    assert pretrained_logits is not None
    assert_shape("pretrained_logits", pretrained_logits, (B, 1, V))

    # Validate that the logits are the same from both models
    assert torch.allclose(logits, pretrained_logits)
