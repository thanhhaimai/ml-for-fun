import random
from dataclasses import dataclass

import tiktoken
import torch

from learning.gpt2.model import GPT2


def assert_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    if tensor.shape != shape:
        raise ValueError(f"Invalid shape: {name}={tensor.shape}, expected: {shape=}")


@dataclass
class HeadAnalysisResult:
    """Results from analyzing a specific attention head"""

    original_probs: torch.Tensor
    patched_probs: torch.Tensor
    kl_divergence: float
    js_divergence: float
    total_variation: float
    l2_distance: float
    cosine_similarity: float
    hellinger_distance: float
    bhattacharyya_coefficient: float
    s1_prob_change: float
    s2_prob_change: float
    logit_diff_change: float


class NameSampler:
    def __init__(self, names: list[str]):
        self.names = names

    def sample(self, num_names: int) -> list[str]:
        return random.sample(self.names, num_names)


class PromptTemplate:
    def __init__(self, template: str, name_sampler: NameSampler):
        self.template = template
        self.name_sampler = name_sampler

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


@dataclass
class TopKProbsResult:
    top_probs: torch.Tensor
    top_indices: torch.Tensor


@dataclass
class CapturedOutput:
    # Mean of the head outputs
    # indexed by block_idx and head_idx
    head_outputs: list[list[torch.Tensor]]

    # Mean of the final probs
    probs: torch.Tensor


class IoiCircuitAnalyzer:
    def __init__(
        self,
        model: GPT2,
        tokenizer: tiktoken.Encoding,
        prompt_template: PromptTemplate,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.device = device

        self.baseline_output: CapturedOutput | None = None

        self.V = self.model.config.vocab_size
        self.H = self.model.config.embedding_size // self.model.config.num_heads

    def topk_probs(self, prompt: str, k: int) -> TopKProbsResult:
        indices = [self.tokenizer.encode(prompt)]
        probs = self.forward(indices)
        assert_shape("probs", probs, (1, self.V))

        top_probs, top_indices = torch.topk(probs.squeeze(0), k=k)
        assert_shape("top_probs", top_probs, (k,))
        assert_shape("top_indices", top_indices, (k,))

        return TopKProbsResult(
            top_probs=top_probs,
            top_indices=top_indices,
        )

    @torch.no_grad()
    def forward(self, indices: list[list[int]]):
        self.model.eval()
        B = len(indices)
        S = len(indices[0])

        # shape: [B, S]
        inputs = torch.tensor(indices, dtype=torch.long, device=self.device)
        assert_shape("inputs", inputs, (B, S))

        # shape: [B, S, V]
        outputs = self.model(inputs)
        assert_shape("outputs", outputs, (B, S, self.V))

        # shape: [B, V]
        last_output = outputs[:, -1, :]
        assert_shape("last_output", last_output, (B, self.V))

        # shape: [B, V]
        probs = torch.softmax(last_output, dim=-1)
        assert_shape("probs", probs, (B, self.V))

        return probs

    def capture_baseline_output(self, batch_size: int):
        """
        Runs the model `batch_size` times with the same template (different names)
        and captures the output of all the heads in all the blocks.

        Returns: the mean of the frozen output. List of blocks of heads of tensors.

        Example:
        ```
        [
            [head_output_1, head_output_2, ...],
            [head_output_1, head_output_2, ...],
        ]
        ```
        """

        self.model.set_capture_output(True)
        self.model.set_use_frozen_output(False)
        cases = [self.prompt_template.sample_abc() for _ in range(batch_size)]
        indices = self.tokenizer.encode_batch(cases)

        B = batch_size
        S = len(indices[0])

        # shape: [V]
        probs = self.forward(indices)
        assert_shape("probs", probs, (B, self.V))

        mean_probs = probs.mean(dim=0, keepdim=True)
        assert_shape("mean_probs", mean_probs, (1, self.V))

        results: list[list[torch.Tensor]] = []
        for block_idx in range(self.model.config.num_blocks):
            block_results: list[torch.Tensor] = []
            for head_idx in range(self.model.config.num_heads):
                output = (
                    self.model.blocks[block_idx].attention.heads[head_idx].frozen_output
                )
                assert_shape("output", output, (B, S, self.H))

                # shape: [B, S, H] -> [1, S, H]
                mean_output = output.mean(dim=0, keepdim=True)
                assert_shape("mean_output", mean_output, (1, S, self.H))

                block_results.append(mean_output)

            results.append(block_results)

        self.model.set_capture_output(False)
        self.baseline_output = CapturedOutput(
            head_outputs=results,
            probs=mean_probs,
        )

    def analyze_head(
        self, block_idx: int, head_idx: int, s1: str, s2: str, s3: str
    ) -> HeadAnalysisResult:
        if self.baseline_output is None:
            raise ValueError("Must call `capture_baseline_output` first")

        # Phase B: capture the output before path patching
        self.model.set_capture_output(True)
        self.model.set_use_frozen_output(False)
        indices = [self.tokenizer.encode(self.prompt_template.from_abc(s1, s2, s3))]

        S = len(indices[0])

        original_probs = self.forward(indices)
        assert_shape("original_probs", original_probs, (1, self.V))

        # Phase C: path patching head[block_idx][head_idx] using the baseline output
        self.model.set_capture_output(False)
        self.model.set_use_frozen_output(True)
        captured_output = self.baseline_output.head_outputs[block_idx][head_idx]
        assert_shape("captured_output", captured_output, (1, S, self.H))

        self.model.blocks[block_idx].attention.heads[
            head_idx
        ].frozen_output = captured_output

        patched_probs = self.forward(indices)
        assert_shape("patched_probs", patched_probs, (1, self.V))

        # Analysis is done, reset `use_frozen_output` to False
        self.model.set_use_frozen_output(False)

        # Flatten for metric computation
        orig_flat = original_probs.squeeze(0)
        patch_flat = patched_probs.squeeze(0)

        # Compute all metrics
        kl_div = self._kl_divergence(orig_flat, patch_flat)
        js_div = self._js_divergence(orig_flat, patch_flat)
        tv_dist = self._total_variation_distance(orig_flat, patch_flat)
        l2_dist = self._l2_distance(orig_flat, patch_flat)
        cos_sim = self._cosine_similarity(orig_flat, patch_flat)
        hellinger_dist = self._hellinger_distance(orig_flat, patch_flat)
        bhatt_coeff = self._bhattacharyya_coefficient(orig_flat, patch_flat)

        s1_prob_change, s2_prob_change, logit_diff_change = self._compute_ioi_metrics(
            orig_flat, patch_flat, s1, s2
        )

        return HeadAnalysisResult(
            original_probs=original_probs,
            patched_probs=patched_probs,
            kl_divergence=kl_div,
            js_divergence=js_div,
            total_variation=tv_dist,
            l2_distance=l2_dist,
            cosine_similarity=cos_sim,
            hellinger_distance=hellinger_dist,
            bhattacharyya_coefficient=bhatt_coeff,
            s1_prob_change=s1_prob_change,
            s2_prob_change=s2_prob_change,
            logit_diff_change=logit_diff_change,
        )

    def _kl_divergence(
        self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8
    ) -> float:
        """
        Kullback-Leibler divergence: KL(p || q) = Σ p(x) * log(p(x) / q(x))

        Measures how much information is lost when using q to approximate p.
        In our context:
        - p = original probability distribution (before head patching)
        - q = patched probability distribution (after head patching)

        Interpretation:
        - KL = 0: Head has no effect (distributions identical)
        - KL ≈ 0.01: Small effect (slight probability changes)
        - KL ≈ 0.1: Moderate effect (noticeable probability changes)
        - KL ≈ 1.0: Large effect (major probability redistribution)
        - KL > 2.0: Very large effect (dramatic change in model behavior)

        Why KL divergence for mechanistic interpretability:
        1. Measures distributional similarity (not just point estimates)
        2. Sensitive to changes in high-probability tokens
        3. Theoretically grounded (information theory)
        4. Widely used standard in the field

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]
            eps: Small constant to avoid log(0)

        Returns:
            KL divergence value (higher = more impact from this head)
        """
        p_safe = torch.clamp(p, min=eps)
        q_safe = torch.clamp(q, min=eps)
        return torch.sum(p_safe * torch.log(p_safe / q_safe)).item()

    def _js_divergence(
        self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8
    ) -> float:
        """
        Jensen-Shannon divergence: JS(p || q) = 0.5 * KL(p || M) + 0.5 * KL(q || M)
        where M = 0.5 * (p + q) is the average distribution.

        A symmetric, bounded version of KL divergence that measures distributional similarity.
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

        Advantages over KL divergence:
        1. Symmetric: JS(p||q) = JS(q||p)
        2. Bounded: Always between 0 and log(2)
        3. More stable: No infinite values
        4. √JS is a true distance metric

        When to prefer JS over KL:
        - When you want symmetric comparison
        - When dealing with sparse distributions (less prone to infinity)
        - When you need bounded, interpretable values
        - For clustering or when metric properties matter

        When to prefer KL over JS:
        - Standard in mechanistic interpretability literature
        - More sensitive to distributional changes
        - When directionality matters (P→Q vs Q→P)

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]
            eps: Small constant to avoid log(0)

        Returns:
            JS divergence value in [0, log(2)], higher = more impact from this head
        """
        p_safe = torch.clamp(p, min=eps)
        q_safe = torch.clamp(q, min=eps)
        m = 0.5 * (p_safe + q_safe)
        kl_pm = torch.sum(p_safe * torch.log(p_safe / m))
        kl_qm = torch.sum(q_safe * torch.log(q_safe / m))
        return (0.5 * (kl_pm + kl_qm)).item()

    def _total_variation_distance(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        Total Variation distance: TV(p, q) = 0.5 * Σ |p(x) - q(x)|

        Measures the minimum amount of probability mass that must be moved to
        transform distribution p into distribution q. Has the most intuitive
        geometric interpretation of all distributional distances.

        In our context:
        - p = original probability distribution (before head patching)
        - q = patched probability distribution (after head patching)

        Interpretation:
        - TV = 0: Distributions identical (head has no effect)
        - TV = 0.1: Need to move 10% of probability mass
        - TV = 0.2: Need to move 20% of probability mass
        - TV = 0.5: Need to move 50% of probability mass (large change)
        - TV = 1.0: Distributions have no overlap (maximum possible change)

        Intuitive meaning:
        If you imagine probability as "sand" distributed across vocabulary tokens,
        TV tells you what fraction of the sand you need to shovel from one pile
        to another to transform the original distribution into the patched one.

        Why TV (0.5 × L1) instead of raw L1 distance:
        1. Bounded [0,1] vs L1's [0,2] for probability distributions
        2. TV = 1 means "completely different" (intuitive)
        3. Standard mathematical definition in probability theory
        4. Cleaner probabilistic interpretation

        Advantages:
        1. Most intuitive interpretation (percentage of probability moved)
        2. Bounded [0,1] with clear meaning
        3. Symmetric: TV(p,q) = TV(q,p)
        4. True metric (satisfies triangle inequality)
        5. Less sensitive to changes in low-probability tokens

        Compared to other metrics:
        - vs KL: More intuitive, less sensitive, bounded
        - vs JS: Simpler formula, equally bounded and symmetric
        - vs L2: Better suited for probability distributions
        - vs L1: Same ranking, but TV has better [0,1] bounds

        When to prefer TV:
        - Want intuitive "percentage moved" interpretation
        - Presenting results to non-technical audiences
        - Care more about where probability mass goes than information theory
        - Working with very sparse distributions

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]

        Returns:
            Total variation distance in [0,1], higher = more probability mass moved
        """
        return (0.5 * torch.sum(torch.abs(p - q))).item()

    def _l1_distance(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        L1 (Manhattan) distance: L1(p, q) = Σ |p(x) - q(x)|

        Raw sum of absolute differences between probability distributions.
        Note: Total Variation = 0.5 × L1, so rankings will be identical.

        Interpretation for probability distributions:
        - L1 = 0: Distributions identical
        - L1 = 0.2: Sum of absolute differences is 0.2
        - L1 = 1.0: Moderate change
        - L1 = 2.0: Maximum possible change (no overlap)

        Why Total Variation is usually preferred over raw L1:
        - TV bounded [0,1] vs L1 bounded [0,2]
        - TV = 1 means "completely different" (intuitive)
        - TV is the standard definition in probability theory

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]

        Returns:
            L1 distance in [0,2], higher = more different (same ranking as TV)
        """
        return torch.sum(torch.abs(p - q)).item()

    def _l2_distance(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        L2 (Euclidean) distance: L2(p, q) = √(Σ (p(x) - q(x))²)

        The "straight line" distance between two probability distributions when
        viewed as points in high-dimensional space. Most familiar distance metric.

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

        Key characteristics:
        1. Squared differences emphasize large changes over small ones
        2. Less sensitive to many small changes than L1/TV
        3. More sensitive to few large changes than L1/TV
        4. Familiar geometric interpretation

        Example sensitivity comparison:
        - Many small changes: L1/TV > L2 (L1/TV more sensitive)
        - Few large changes: L2 > L1/TV (L2 more sensitive)

        Advantages:
        1. Most familiar distance metric (Euclidean)
        2. Differentiable everywhere (useful for optimization)
        3. Less sensitive to outliers than higher norms
        4. Natural geometric interpretation
        5. Symmetric and satisfies triangle inequality

        Disadvantages:
        1. Less intuitive interpretation than TV for probability distributions
        2. Can be dominated by a few large changes
        3. Not as standard in probability/information theory as KL/JS
        4. Bounded [0,√2] is less intuitive than TV's [0,1]

        Compared to other metrics:
        - vs KL: Less sensitive, no information-theoretic meaning, bounded
        - vs JS: Similar boundedness, but less probability-theoretic
        - vs TV/L1: Emphasizes large changes more, less intuitive for probabilities
        - vs higher Lp norms: More robust, less dominated by outliers

        When to prefer L2:
        - When you care more about large probability changes than many small ones
        - Familiar with Euclidean distance concepts
        - Need differentiability (for gradient-based methods)
        - Want to downweight the impact of many tiny changes

        When to avoid L2:
        - Want to count all probability movement equally (use TV)
        - Need information-theoretic interpretation (use KL/JS)
        - Working with very sparse distributions
        - Want most intuitive probability interpretation

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]

        Returns:
            L2 distance in [0, √2], higher = more different (emphasizes large changes)
        """
        return torch.norm(p - q, p=2).item()

    def _cosine_similarity(
        self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8
    ) -> float:
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

        Example where they differ:
        P = [0.8, 0.15, 0.05]  # Strong preference for token 1
        Q = [0.4, 0.075, 0.025] # Same relative pattern, half the "confidence"

        - High cosine similarity (≈1.0): Same relative preferences
        - High distance metrics: Lots of probability moved

        This reveals different types of head behavior:
        - High cos_sim + high distance = Head changes confidence but preserves preferences
        - Low cos_sim + high distance = Head fundamentally changes token preferences
        - Low cos_sim + low distance = Head makes subtle but important pattern shifts

        When cosine similarity is most useful:
        1. Understanding if heads preserve relative token importance
        2. Detecting "confidence adjustment" vs "preference changing" heads
        3. Finding heads that maintain distributional shape
        4. Comparing distribution patterns regardless of magnitude

        Advantages:
        1. Scale-invariant (focuses on shape, not magnitude)
        2. Intuitive geometric interpretation (angle between vectors)
        3. Reveals different information than distance metrics
        4. Good for understanding relative importance preservation

        Disadvantages:
        1. Less sensitive to magnitude changes (which may be important)
        2. Can be misleading if you care about absolute probability values
        3. Less standard in mechanistic interpretability literature
        4. Harder to interpret for sparse distributions

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]
            eps: Small constant for numerical stability

        Returns:
            Cosine similarity in [0,1], higher = more similar relative patterns
        """
        # Add small epsilon for numerical stability
        p_safe = p + eps
        q_safe = q + eps

        dot_product = torch.sum(p_safe * q_safe)
        norm_p = torch.norm(p_safe, p=2)
        norm_q = torch.norm(q_safe, p=2)

        return (dot_product / (norm_p * norm_q)).item()

    def _hellinger_distance(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        Hellinger distance: H(P,Q) = (1/√2) × √Σ(√P(x) - √Q(x))²

        A metric specifically designed for probability distributions, based on
        the Hellinger coefficient. Popular in statistics and probability theory.

        Properties:
        - Bounded [0,1]
        - Symmetric
        - More robust to outliers than L2
        - Less sensitive to differences in small probabilities

        Interpretation:
        - H = 0: Identical distributions
        - H ≈ 0.1: Small difference
        - H ≈ 0.3: Moderate difference
        - H ≈ 0.7: Large difference
        - H = 1: Completely disjoint distributions

        When to use:
        - Want a probability-specific metric
        - More robust alternative to L2
        - Working with sparse distributions
        - Need bounded [0,1] metric that's probability-theoretic

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]

        Returns:
            Hellinger distance in [0,1], higher = more different
        """
        sqrt_p = torch.sqrt(p)
        sqrt_q = torch.sqrt(q)
        return (
            (1 / torch.sqrt(torch.tensor(2.0))) * torch.norm(sqrt_p - sqrt_q, p=2)
        ).item()

    def _bhattacharyya_coefficient(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        Bhattacharyya coefficient: BC(P,Q) = Σ √(P(x) × Q(x))

        Measures the amount of overlap between two probability distributions.
        Related to Hellinger distance: H² = 1 - BC

        Properties:
        - Range [0,1]
        - 1 = identical distributions
        - 0 = no overlap
        - Geometric mean of probabilities

        Interpretation:
        - BC = 1.0: Perfect overlap (identical)
        - BC ≈ 0.9: High overlap
        - BC ≈ 0.7: Moderate overlap
        - BC ≈ 0.3: Low overlap
        - BC = 0.0: No overlap (disjoint)

        When to use:
        - Want to measure distributional "overlap"
        - Interested in shared probability mass
        - Complementary to distance metrics

        Args:
            p: Original probability distribution [vocab_size]
            q: Patched probability distribution [vocab_size]

        Returns:
            Bhattacharyya coefficient in [0,1], higher = more overlap
        """
        return torch.sum(torch.sqrt(p * q)).item()

    def _compute_ioi_metrics(
        self,
        original_probs: torch.Tensor,
        patched_probs: torch.Tensor,
        s1: str,
        s2: str,
    ) -> tuple[float, float, float]:
        """
        Compute IOI-specific metrics

        `s1` and `s2` are the names of the two subjects.
        """
        # Get token indices (with space padding)
        s1_idx = self.tokenizer.encode(f" {s1}")[0]
        s2_idx = self.tokenizer.encode(f" {s2}")[0]

        # Probability changes for specific tokens
        s1_prob_change = (original_probs[s1_idx] - patched_probs[s1_idx]).item()
        s2_prob_change = (original_probs[s2_idx] - patched_probs[s2_idx]).item()

        # Compute logit differences
        original_logit_diff = torch.log(original_probs[s1_idx] / original_probs[s2_idx])
        patched_logit_diff = torch.log(patched_probs[s1_idx] / patched_probs[s2_idx])
        logit_diff_change = (original_logit_diff - patched_logit_diff).item()

        return s1_prob_change, s2_prob_change, logit_diff_change
