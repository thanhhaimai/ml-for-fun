import pytest
import tiktoken
import torch

from learning.ioi_circuit.data_sources import NamesDataSource
from learning.ioi_circuit.ioi_circuit_analyzer import (
    IoiCircuitAnalyzer,
    PathPatchingConfig,
)
from learning.ioi_circuit.model import GPT2, HeadId, ModelConfig, PretrainedName
from learning.ioi_circuit.prompts import PromptTemplate


@pytest.fixture
def model() -> GPT2:
    model_config = ModelConfig(
        num_blocks=4,
        num_heads=2,
        embedding_size=4,
        sequence_length=16,
    )
    model = GPT2(model_config)
    model.eval()
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
def prompt_template(
    data_source: NamesDataSource, device: torch.device
) -> PromptTemplate:
    return PromptTemplate(
        template="When{s1} and{s2} went to the store,{s3} gave a drink to",
        names_data_source=data_source,
        device=device,
    )


@pytest.fixture
def analyzer(
    model: GPT2,
    tokenizer: tiktoken.Encoding,
    prompt_template: PromptTemplate,
    device: torch.device,
) -> IoiCircuitAnalyzer:
    return IoiCircuitAnalyzer(model, tokenizer, prompt_template, device)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_topk_logits(
    tokenizer: tiktoken.Encoding,
    data_source: NamesDataSource,
    device: torch.device,
):
    model, pretrained_model = GPT2.from_pretrained(
        PretrainedName.GPT2_SMALL,
        device=torch.device("cpu"),
    )

    model.eval()

    prompt_template = PromptTemplate(
        template="When{s1} and{s2} went to the store,{s3} gave a drink to",
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


@torch.no_grad()
def test_capture_output(
    analyzer: IoiCircuitAnalyzer,
    prompt_template: PromptTemplate,
):
    BATCH_SIZE = 2
    baseline_batch = prompt_template.sample_batch_abc(BATCH_SIZE)
    baseline_output = analyzer.capture_baseline_output(baseline_batch.prompts)

    for block_idx in range(analyzer.model.config.num_blocks):
        for head_idx in range(analyzer.model.config.num_heads):
            frozen_output = analyzer.model.get_head(
                HeadId(block_idx, head_idx)
            ).frozen_output
            assert torch.allclose(
                baseline_output.head_outputs[block_idx][head_idx], frozen_output
            )


@torch.no_grad()
def test_path_patching(
    analyzer: IoiCircuitAnalyzer,
    prompt_template: PromptTemplate,
):
    BATCH_SIZE = 1
    baseline_batch = prompt_template.sample_batch_abc(BATCH_SIZE)
    baseline_output = analyzer.capture_baseline_output(baseline_batch.prompts)

    prompts_abb = prompt_template.sample_batch_abb(BATCH_SIZE)
    prepatched_output = analyzer.capture_baseline_output(prompts_abb.prompts)

    logits_prepatched, logits_patched = analyzer.path_patching(
        PathPatchingConfig(
            start_head=HeadId(0, 1),
            end_heads=[
                HeadId(2, 0),
                HeadId(3, 1),
            ],
        ),
        baseline_output,
        prompts_abb.prompts,
    )

    assert logits_prepatched.shape == (BATCH_SIZE, analyzer.model.vocab_size)
    assert logits_patched.shape == (BATCH_SIZE, analyzer.model.vocab_size)

    # These heads are unchanged by the path patching
    for head_id in [
        HeadId(0, 0),
        HeadId(0, 1),
        HeadId(1, 0),
        HeadId(1, 1),
        HeadId(2, 1),
    ]:
        assert torch.allclose(
            analyzer.model.get_head(head_id).frozen_output,
            prepatched_output.head_outputs[head_id.block_idx][head_id.head_idx],
        )

    # Head 2.0 and 3.1 will be recomputed using the frozen input
    # Head 3.0 will be recomputed because it's affected by head 2.0
    for head_id in [
        HeadId(2, 0),
        HeadId(3, 1),
        HeadId(3, 0),
    ]:
        print(head_id)
        assert not torch.allclose(
            analyzer.model.get_head(head_id).frozen_output,
            prepatched_output.head_outputs[head_id.block_idx][head_id.head_idx],
        )

    # Check that the logits are different
    assert not torch.allclose(logits_patched, logits_prepatched)
    assert not torch.allclose(logits_patched, baseline_output.logits)
    assert not torch.allclose(logits_patched, prepatched_output.logits)
