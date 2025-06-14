from dataclasses import dataclass

import torch

from learning.ioi_circuit.data_sources import NamesDataSource

PLACES = [
    " park",
    " restaurant",
    " library",
    " museum",
    " zoo",
    " beach",
    " mall",
    " theater",
]


OBJECTS = [
    " book",
    " notebook",
    " toy",
    " gift",
    " card",
    " letter",
    " drink",
    " pen",
    " pencil",
]

# NOTE: there is no space in front of each slot because the token already includes a space
IOI_TEMPLATES = [
    "Then,{s1} and{s2} went to the{place}.{s3} gave a{object} to ",
    "Then,{s1} and{s2} had a lot of fun at the{place}.{s3} gave a{object} to ",
    "Then,{s1} and{s2} were working at the{place}.{s3} decided to give a{object} to ",
    "Then,{s1} and{s2} were thinking about going to the{place}.{s3} wanted to give a{object} to ",
    "Then,{s1} and{s2} had a long argument, and afterwards{s3} said to ",
    "After{s1} and{s2} went to the{place},{s3} gave a{object} to ",
    "When{s1} and{s2} got a{object} at the{place},{s3} decided to give it to ",
    "When{s1} and{s2} got a{object} at the{place},{s3} decided to give the [OBJECT] to ",
    "While{s1} and{s2} were working at the{place},{s3} gave a{object} to ",
    "While{s1} and{s2} were commuting to the{place},{s3} gave a{object} to ",
    "After the lunch,{s1} and{s2} went to the{place}.{s3} gave a{object} to ",
    "Afterwards,{s1} and{s2} went to the{place}.{s3} gave a{object} to ",
    "Then,{s1} and{s2} had a long argument. Afterwards{s3} said to ",
    "The{place}{s1} and{s2} went to had a{object}.{s3} gave it to ",
    "Friends{s1} and{s2} found a{object} at the{place}.{s3} gave it to ",
]


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
    def __init__(
        self,
        template: str,
        names_data_source: NamesDataSource,
        device: torch.device,
    ):
        self.template = template
        self.names_data_source = names_data_source
        self.device = device

    def sample_batch_abc(self, batch_size: int) -> PromptBatch:
        name_samples = self.names_data_source.sample_batch(3, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        s3_indices = []
        for name_sample in name_samples:
            s1, s2, s3 = name_sample.names_with_space
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
        name_samples = self.names_data_source.sample_batch(2, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        for name_sample in name_samples:
            s1, s2 = name_sample.names_with_space
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
        name_samples = self.names_data_source.sample_batch(2, batch_size)
        prompts = []
        s1_indices = []
        s2_indices = []
        for name_sample in name_samples:
            s1, s2 = name_sample.names_with_space
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
        name_sample = self.names_data_source.sample(3)
        s1, s2, s3 = name_sample.names_with_space
        return self.template.format(s1=s1, s2=s2, s3=s3)

    def sample_aba(self) -> str:
        name_sample = self.names_data_source.sample(2)
        s1, s2 = name_sample.names_with_space
        return self.template.format(s1=s1, s2=s2, s3=s1)

    def sample_abb(self) -> str:
        name_sample = self.names_data_source.sample(2)
        s1, s2 = name_sample.names_with_space
        return self.template.format(s1=s1, s2=s2, s3=s2)

    def from_abc(self, s1, s2, s3) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s3)

    def from_aba(self, s1, s2) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s1)

    def from_abb(self, s1, s2) -> str:
        return self.template.format(s1=s1, s2=s2, s3=s2)
