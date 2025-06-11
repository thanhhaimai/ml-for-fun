import random
from dataclasses import dataclass
from typing import NamedTuple

import tiktoken
import torch


class NameSample(NamedTuple):
    names: list[str]
    indices: list[int]


class NameSampler:
    def __init__(self, names: list[str], tokenizer: tiktoken.Encoding):
        self.names = names
        self.indices = [tokenizer.encode(f" {name.strip()}")[0] for name in names]

    def sample(self, num_names: int) -> list[str]:
        return random.sample(self.names, num_names)

    def sample_batch(self, num_names: int, batch_size: int) -> list[NameSample]:
        batches = []
        for _ in range(batch_size):
            sample_indices = random.sample(range(len(self.indices)), num_names)
            names = [self.names[i] for i in sample_indices]
            indices = [self.indices[i] for i in sample_indices]

            batches.append(NameSample(names, indices))

        return batches


@dataclass
class PromptBatch:
    # The prompts uses s1, s2, s3 as placeholders
    # The indices are the indices of the s1, s2, s3 tokens
    prompts: list[str]
    # shape: [B]
    s1_indices: torch.Tensor
    # shape: [B]
    s2_indices: torch.Tensor
    # shape: [B]
    s3_indices: torch.Tensor


class PromptTemplate:
    def __init__(self, template: str, name_sampler: NameSampler, device: torch.device):
        self.template = template
        self.name_sampler = name_sampler
        self.device = device

    def sample_batch_abc(self, batch_size: int) -> PromptBatch:
        name_samples = self.name_sampler.sample_batch(3, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        s3_indices = []
        for name_sample in name_samples:
            s1, s2, s3 = name_sample.names
            prompts.append(self.template.format(s1=s1, s2=s2, s3=s3))
            s1_indices.append(name_sample.indices[0])
            s2_indices.append(name_sample.indices[1])
            s3_indices.append(name_sample.indices[2])

        return PromptBatch(
            prompts,
            torch.tensor(s1_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
            torch.tensor(s3_indices, device=self.device),
        )

    def sample_batch_aba(self, batch_size: int) -> PromptBatch:
        name_samples = self.name_sampler.sample_batch(2, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        for name_sample in name_samples:
            s1, s2 = name_sample.names
            prompts.append(self.template.format(s1=s1, s2=s2, s3=s1))
            s1_indices.append(name_sample.indices[0])
            s2_indices.append(name_sample.indices[1])

        return PromptBatch(
            prompts,
            torch.tensor(s1_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
            torch.tensor(s1_indices, device=self.device),
        )

    def sample_batch_abb(self, batch_size: int) -> PromptBatch:
        name_samples = self.name_sampler.sample_batch(2, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        for name_sample in name_samples:
            s1, s2 = name_sample.names
            prompts.append(self.template.format(s1=s1, s2=s2, s3=s2))
            s1_indices.append(name_sample.indices[0])
            s2_indices.append(name_sample.indices[1])

        return PromptBatch(
            prompts,
            torch.tensor(s1_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
            torch.tensor(s2_indices, device=self.device),
        )

    def sample_abc(self) -> str:
        s1, s2, s3 = self.name_sampler.sample(3)
        return self.template.format(s1=s1, s2=s2, s3=s3)

    def sample_aba(self) -> str:
        s1, s2 = self.name_sampler.sample(2)
        return self.template.format(s1=s1, s2=s2, s3=s1)

    def sample_abb(self) -> str:
        s1, s2 = self.name_sampler.sample(2)
        return self.template.format(s1=s1, s2=s2, s3=s2)

    def from_abc(self, s1, s2, s3) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s3)

    def from_aba(self, s1, s2) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s1)

    def from_abb(self, s1, s2) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s2)
