from dataclasses import dataclass
from typing import Self

import pandas as pd
import torch


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Kullback-Leibler divergence: KL(p || q) = Σ p(x) * log(p(x) / q(x))

    Measures how much information is lost when using q to approximate p.
    Prefer this over JS divergence when directionality matters.

    In our context:
    - p = original probability distribution (before head patching)
    - q = patched probability distribution (after head patching)

    Interpretation:
    - KL = 0: Head has no effect (distributions identical)
    - KL ≈ 0.01: Small effect (slight probability changes)
    - KL ≈ 0.1: Moderate effect (noticeable probability changes)
    - KL ≈ 1.0: Large effect (major probability redistribution)
    - KL > 2.0: Very large effect (dramatic change in model behavior)
    """
    B, V = p.shape
    assert_shape("p", p, (B, V))
    assert_shape("q", q, (B, V))

    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    kl = torch.sum(p_safe * torch.log(p_safe / q_safe), dim=-1)
    assert_shape("kl", kl, (B,))

    return kl.mean().item()


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Jensen-Shannon divergence: JS(p || q) = 0.5 * KL(p || M) + 0.5 * KL(q || M)
    where M = 0.5 * (p + q) is the average distribution.

    A symmetric, bounded version of KL divergence that measures distributional similarity.
    Prefer this over KL divergence when directionality does not matter.

    In our context:
    - p = original probability distribution (before head patching)
    - q = patched probability distribution (after head patching)

    Interpretation:
    - JS = 0: Distributions are identical (head has no effect)
    - JS ≈ 0.01: Small distributional change
    - JS ≈ 0.05: Moderate distributional change
    - JS ≈ 0.1: Large distributional change
    - JS ≈ 0.3: Very large distributional change
    - JS approaches log(2) ≈ 0.693: Maximum possible divergence
    """
    B, V = p.shape
    assert_shape("p", p, (B, V))
    assert_shape("q", q, (B, V))

    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    m = 0.5 * (p_safe + q_safe)
    assert_shape("m", m, (B, V))

    kl_pm = torch.sum(p_safe * torch.log(p_safe / m), dim=-1)
    kl_qm = torch.sum(q_safe * torch.log(q_safe / m), dim=-1)
    assert_shape("kl_pm", kl_pm, (B,))
    assert_shape("kl_qm", kl_qm, (B,))

    return (0.5 * (kl_pm + kl_qm)).mean().item()


def _total_variation_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Total Variation distance: TV(p, q) = 0.5 * Σ |p(x) - q(x)|

    Measures the minimum amount of probability mass that must be moved to
    transform distribution p into distribution q.

    In our context:
    - p = original probability distribution (before head patching)
    - q = patched probability distribution (after head patching)

    Interpretation:
    - TV = 0: Distributions identical (head has no effect)
    - TV = 0.1: Need to move 10% of probability mass
    - TV = 0.2: Need to move 20% of probability mass
    - TV = 0.5: Need to move 50% of probability mass (large change)
    - TV = 1.0: Distributions have no overlap (maximum possible change)
    """
    B, V = p.shape
    assert_shape("p", p, (B, V))
    assert_shape("q", q, (B, V))

    tv = 0.5 * torch.sum(torch.abs(p - q), dim=-1)
    assert_shape("tv", tv, (B,))

    return tv.mean().item()


def _l2_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    L2 (Euclidean) distance: L2(p, q) = √(Σ (p(x) - q(x))²)

    The "straight line" distance between two probability distributions when
    viewed as points in high-dimensional space.

    In our context:
    - p = original probability distribution (before head patching)
    - q = patched probability distribution (after head patching)

    Interpretation:
    - L2 = 0: Distributions identical (head has no effect)
    - L2 ≈ 0.1: Small distributional change
    - L2 ≈ 0.3: Moderate distributional change
    - L2 ≈ 0.7: Large distributional change
    - L2 ≈ 1.4: Very large distributional change
    - L2 = √2 ≈ 1.414: Maximum possible distance (no overlap)
    """
    B, V = p.shape
    assert_shape("p", p, (B, V))
    assert_shape("q", q, (B, V))

    l2 = torch.norm(p - q, p=2, dim=-1)
    assert_shape("l2", l2, (B,))
    return l2.mean().item()


def _cosine_similarity(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Cosine similarity: cos(θ) = (P·Q) / (||P|| × ||Q||)

    Measures the cosine of the angle between two probability distributions.
    Focuses on the SHAPE/DIRECTION of distributions rather than absolute differences.

    In our context:
    - p = original probability distribution (before head patching)
    - q = patched probability distribution (after head patching)

    Interpretation:
    - cos_sim = 1.0: Identical relative patterns (perfect shape match)
    - cos_sim ≈ 0.9: Very similar relative patterns
    - cos_sim ≈ 0.7: Moderately similar patterns
    - cos_sim ≈ 0.5: Somewhat different patterns
    - cos_sim ≈ 0.0: Completely different relative patterns (orthogonal)

    Key insight - cosine similarity vs distance metrics:
    Distance metrics ask: "How much probability moved?"
    Cosine similarity asks: "Did the relative importance pattern change?"
    """
    B, V = p.shape
    assert_shape("p", p, (B, V))
    assert_shape("q", q, (B, V))

    dot_product = torch.sum(p * q, dim=-1)
    norm_p = torch.norm(p, p=2, dim=-1)
    norm_q = torch.norm(q, p=2, dim=-1)
    assert_shape("dot_product", dot_product, (B,))
    assert_shape("norm_p", norm_p, (B,))
    assert_shape("norm_q", norm_q, (B,))

    return (dot_product / (norm_p * norm_q)).mean().item()


def _hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Hellinger distance: H(P,Q) = (1/√2) × √Σ(√P(x) - √Q(x))²

    A metric based on the Hellinger coefficient. Less sensitive to differences
    in small probabilities and more sensitive to differences in large probabilities.

    In our context:
    - p = original probability distribution (before head patching)
    - q = patched probability distribution (after head patching)

    Interpretation:
    - H = 0: Identical distributions
    - H ≈ 0.1: Small difference
    - H ≈ 0.3: Moderate difference
    - H ≈ 0.7: Large difference
    - H = 1: Completely disjoint distributions
    """
    B, V = p.shape
    assert_shape("p", p, (B, V))
    assert_shape("q", q, (B, V))

    sqrt_p = torch.sqrt(p)
    sqrt_q = torch.sqrt(q)
    assert_shape("sqrt_p", sqrt_p, (B, V))
    assert_shape("sqrt_q", sqrt_q, (B, V))

    hellinger = (1 / torch.sqrt(torch.tensor(2.0))) * torch.norm(
        sqrt_p - sqrt_q, p=2, dim=-1
    )
    assert_shape("hellinger", hellinger, (B,))

    return hellinger.mean().item()


@dataclass
class DiffLogitsMetrics:
    s1_logit: float
    s2_logit: float
    s1_prob: float
    s2_prob: float

    def summary(self) -> dict[str, float]:
        return {
            "s1_logit": self.s1_logit,
            "s2_logit": self.s2_logit,
            "s1_prob": self.s1_prob,
            "s2_prob": self.s2_prob,
        }

    @classmethod
    def merge_df(cls, metrics: list[Self]) -> pd.DataFrame:
        return pd.DataFrame([m.summary() for m in metrics])


@dataclass
class ProbsMetrics:
    # Logits Metrics
    s1_prob_original: float
    s2_prob_original: float
    s1_prob_patched: float
    s2_prob_patched: float
    s1_prob_factor: float
    s2_prob_factor: float

    # Logits Metrics
    s1_logit_diff: float
    s2_logit_diff: float

    # Probs Metrics
    kl_divergence: float
    js_divergence: float
    total_variation: float
    l2_distance: float
    cosine_similarity: float
    hellinger_distance: float

    @classmethod
    def from_logits(
        cls,
        original_logits: torch.Tensor,
        patched_logits: torch.Tensor,
        s1_indices: torch.Tensor,
        s2_indices: torch.Tensor,
    ) -> Self:
        B, V = original_logits.shape
        assert_shape("original_logits", original_logits, (B, V))
        assert_shape("patched_logits", patched_logits, (B, V))
        s1_indices = s1_indices.unsqueeze(-1)
        s2_indices = s2_indices.unsqueeze(-1)
        assert_shape("s1_indices", s1_indices, (B, 1))
        assert_shape("s2_indices", s2_indices, (B, 1))

        original_probs = torch.softmax(original_logits, dim=-1)
        patched_probs = torch.softmax(patched_logits, dim=-1)
        assert_shape("original_probs", original_probs, (B, V))
        assert_shape("patched_probs", patched_probs, (B, V))

        original_s1_probs = original_probs.gather(dim=-1, index=s1_indices)
        original_s2_probs = original_probs.gather(dim=-1, index=s2_indices)
        patched_s1_probs = patched_probs.gather(dim=-1, index=s1_indices)
        patched_s2_probs = patched_probs.gather(dim=-1, index=s2_indices)
        assert_shape("original_s1_probs", original_s1_probs, (B, 1))
        assert_shape("original_s2_probs", original_s2_probs, (B, 1))
        assert_shape("patched_s1_probs", patched_s1_probs, (B, 1))
        assert_shape("patched_s2_probs", patched_s2_probs, (B, 1))

        probs_factor = patched_probs / original_probs
        s1_prob_factor = probs_factor.gather(dim=-1, index=s1_indices)
        s2_prob_factor = probs_factor.gather(dim=-1, index=s2_indices)
        assert_shape("s1_prob_factor", s1_prob_factor, (B, 1))
        assert_shape("s2_prob_factor", s2_prob_factor, (B, 1))

        diff_logits = patched_logits - original_logits
        s1_logit_diff = diff_logits.gather(dim=-1, index=s1_indices)
        s2_logit_diff = diff_logits.gather(dim=-1, index=s2_indices)
        assert_shape("s1_logit_diff", s1_logit_diff, (B, 1))
        assert_shape("s2_logit_diff", s2_logit_diff, (B, 1))

        kl_divergence = _kl_divergence(original_probs, patched_probs)
        js_divergence = _js_divergence(original_probs, patched_probs)
        total_variation = _total_variation_distance(original_probs, patched_probs)
        l2_distance = _l2_distance(original_probs, patched_probs)
        cosine_similarity = _cosine_similarity(original_probs, patched_probs)
        hellinger_distance = _hellinger_distance(original_probs, patched_probs)

        return cls(
            s1_prob_original=original_s1_probs.mean().item(),
            s2_prob_original=original_s2_probs.mean().item(),
            s1_prob_patched=patched_s1_probs.mean().item(),
            s2_prob_patched=patched_s2_probs.mean().item(),
            s1_prob_factor=s1_prob_factor.mean().item(),
            s2_prob_factor=s2_prob_factor.mean().item(),
            s1_logit_diff=s1_logit_diff.mean().item(),
            s2_logit_diff=s2_logit_diff.mean().item(),
            kl_divergence=kl_divergence,
            js_divergence=js_divergence,
            total_variation=total_variation,
            l2_distance=l2_distance,
            cosine_similarity=cosine_similarity,
            hellinger_distance=hellinger_distance,
        )

    def summary(self) -> dict[str, float]:
        return {
            "KL": self.kl_divergence,
            "JS": self.js_divergence,
            "TV": self.total_variation,
            "L2": self.l2_distance,
            "cos_sim": self.cosine_similarity,
            "Hellinger": self.hellinger_distance,
            "s1_prob_original": self.s1_prob_original,
            "s1_prob_patched": self.s1_prob_patched,
            "s1_prob_factor": self.s1_prob_factor,
            "s1_logit_diff": self.s1_logit_diff,
            "s2_prob_original": self.s2_prob_original,
            "s2_prob_patched": self.s2_prob_patched,
            "s2_prob_factor": self.s2_prob_factor,
            "s2_logit_diff": self.s2_logit_diff,
        }

    @classmethod
    def merge_df(cls, metrics: list[Self]) -> pd.DataFrame:
        return pd.DataFrame([m.summary() for m in metrics])
