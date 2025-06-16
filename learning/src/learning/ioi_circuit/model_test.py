import pytest
import tiktoken
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from learning.ioi_circuit.model import (
    GPT2,
    PRETRAINED_CONFIG,
    HeadId,
    PretrainedName,
)


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@torch.no_grad()
def test_all_layers_output(tokenizer: tiktoken.Encoding):
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
def test_same_pretrained_logits(tokenizer: tiktoken.Encoding):
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


def test_set_mode_single_head():
    model = GPT2(PRETRAINED_CONFIG[PretrainedName.TEST])
    model.eval()

    model.set_mode(
        capture_input=[HeadId(0, 0)],
        use_frozen_input=[HeadId(0, 1)],
        capture_output=[HeadId(1, 0)],
        use_frozen_output=[HeadId(1, 1)],
    )

    for layer_idx in range(model.config.num_blocks):
        for head_idx in range(model.config.num_heads):
            head_id = HeadId(layer_idx, head_idx)
            if head_id == HeadId(0, 0):
                assert model.get_head(head_id).should_capture_input
            else:
                assert not model.get_head(head_id).should_capture_input

            if head_id == HeadId(0, 1):
                assert model.get_head(head_id).use_frozen_input
            else:
                assert not model.get_head(head_id).use_frozen_input

            if head_id == HeadId(1, 0):
                assert model.get_head(head_id).should_capture_output
            else:
                assert not model.get_head(head_id).should_capture_output

            if head_id == HeadId(1, 1):
                assert model.get_head(head_id).use_frozen_output
            else:
                assert not model.get_head(head_id).use_frozen_output


def test_mode_flip_flop():
    model = GPT2(PRETRAINED_CONFIG[PretrainedName.TEST])
    model.eval()

    # Enable all modes
    model.set_mode(
        capture_input=True,
        use_frozen_input=True,
        capture_output=True,
        use_frozen_output=True,
    )

    # Only set modes for head (1, 1)
    # This should disable all other heads
    model.set_mode(
        capture_input=[HeadId(1, 1)],
        use_frozen_input=[HeadId(1, 1)],
        capture_output=[HeadId(1, 1)],
        use_frozen_output=[HeadId(1, 1)],
    )

    # Verify that all heads are disabled
    for layer_idx in range(model.config.num_blocks):
        for head_idx in range(model.config.num_heads):
            head_id = HeadId(layer_idx, head_idx)
            if head_id == HeadId(1, 1):
                assert model.get_head(head_id).should_capture_input
                assert model.get_head(head_id).use_frozen_input
                assert model.get_head(head_id).should_capture_output
                assert model.get_head(head_id).use_frozen_output
            else:
                assert not model.get_head(head_id).should_capture_input
                assert not model.get_head(head_id).use_frozen_input
                assert not model.get_head(head_id).should_capture_output
                assert not model.get_head(head_id).use_frozen_output


@pytest.mark.parametrize("capture_input", [True, False])
@pytest.mark.parametrize("use_frozen_input", [True, False])
@pytest.mark.parametrize("capture_output", [True, False])
@pytest.mark.parametrize("use_frozen_output", [True, False])
def test_set_mode(
    capture_input: bool | list[HeadId],
    use_frozen_input: bool | list[HeadId],
    capture_output: bool | list[HeadId],
    use_frozen_output: bool | list[HeadId],
):
    model = GPT2(PRETRAINED_CONFIG[PretrainedName.TEST])
    model.eval()

    model.set_mode(
        capture_input=capture_input,
        use_frozen_input=use_frozen_input,
        capture_output=capture_output,
        use_frozen_output=use_frozen_output,
    )

    assert model.get_head(HeadId(0, 0)).should_capture_input == capture_input
    assert model.get_head(HeadId(0, 0)).use_frozen_input == use_frozen_input
    assert model.get_head(HeadId(0, 0)).should_capture_output == capture_output
    assert model.get_head(HeadId(0, 0)).use_frozen_output == use_frozen_output


@torch.no_grad()
def test_frozen_output(tokenizer: tiktoken.Encoding):
    model = GPT2(PRETRAINED_CONFIG[PretrainedName.TEST])
    model.eval()

    B = 1
    S = 1
    E = model.embedding_size

    text = "Mary"
    embedding_1 = model.get_embedding(torch.tensor([tokenizer.encode(text)]))
    assert_shape("embedding_1", embedding_1, (B, S, E))

    # Test that capturing output saves the output to frozen_output
    model.set_mode(
        capture_input=False,
        use_frozen_input=False,
        capture_output=True,
        use_frozen_output=False,
    )
    output_1 = model.blocks[0].attention.heads[0](embedding_1)
    assert torch.allclose(model.get_head(HeadId(0, 0)).frozen_output, output_1)

    # Test that using frozen output gives the same output as the captured output, even when the input is different
    text = "John"
    embedding_2 = model.get_embedding(torch.tensor([tokenizer.encode(text)]))
    assert_shape("embedding_2", embedding_2, (B, S, E))

    model.set_mode(
        capture_input=False,
        use_frozen_input=False,
        capture_output=False,
        use_frozen_output=True,
    )
    output_2 = model.blocks[0].attention.heads[0](embedding_2)
    assert torch.allclose(output_1, output_2)


@torch.no_grad()
def test_frozen_input(tokenizer: tiktoken.Encoding):
    model = GPT2(PRETRAINED_CONFIG[PretrainedName.TEST])
    model.eval()

    B = 1
    S = 1
    E = model.embedding_size

    text = "Mary"
    embedding_1 = model.get_embedding(torch.tensor([tokenizer.encode(text)]))
    assert_shape("embedding_1", embedding_1, (B, S, E))

    # Test that capturing input saves the input to frozen_input
    model.set_mode(
        capture_input=True,
        use_frozen_input=False,
        capture_output=False,
        use_frozen_output=False,
    )
    output_1 = model.blocks[0].attention.heads[0](embedding_1)
    assert torch.allclose(model.get_head(HeadId(0, 0)).frozen_input, embedding_1)

    # Test that using frozen input gives the same output_1, even when the input is different
    text = "John"
    embedding_2 = model.get_embedding(torch.tensor([tokenizer.encode(text)]))
    assert_shape("embedding_2", embedding_2, (B, S, E))

    model.set_mode(
        capture_input=False,
        use_frozen_input=True,
        capture_output=False,
        use_frozen_output=False,
    )
    output_2 = model.blocks[0].attention.heads[0](embedding_2)
    assert torch.allclose(output_1, output_2)
