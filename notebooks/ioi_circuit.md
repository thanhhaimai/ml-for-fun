# Research Log for IOI Circuit

Reproducting this paper: <https://arxiv.org/abs/2211.00593>

## 1: Reproduce IOI observations from paper

Test 1: The paper claims that GPT2 small would produce "Mary" 68% of time for this sentence:

```python
When Mary and John went to the store, John gave a drink to | seed=42
0.45  Mary
0.21  them
0.07  the
```

Observation: we might not have the same seed, but the claim of prob=0.68 seems high

- Maybe I didn't fully reproduce GPT2 correctly? -- see (3:) for update
- Maybe the GPT2 weights from HF is not 100% the same with the TensorFlow GPT2 version? Not sure if the paper uses pytorch GPT2 or TF GPT2.

Test 2: Slightly changes the sentence (names, store -> park, drink -> leaf)

```python
When Vincent and Vanessa went to the park, Vincent gave a leaf to
0.51  Vanessa
0.12  the
0.06  a
When Vincent and Vanessa went to the park, Vanessa gave a leaf to
0.54  Vincent
0.09  the
0.03  her
```

Observation: Still reproducible

Test 3: Change from 1 sentence to 2 sentences

```python
Mary and John went to the store. John gave a drink to
0.31  them
0.15  John
0.13  the
Mary and John went to the store; John gave a drink to
0.33  them
0.15  John
0.14  the
Mary and John went to the store! John gave a drink to
0.27  them
0.13  the
0.10  John
Mary and John went to the store. Mary gave a drink to
0.46  John
0.14  them
0.12  the
Mary and John went to the store; Mary gave a drink to
0.29  John
0.21  them
0.14  the
Mary and John went to the store! Mary gave a drink to
0.40  John
0.14  the
0.11  them
```

Observation: Doesn't work if it's "John gave", but works with "Mary gave"

- This feels like the original research result might be more fragile than the paper implies -- see (4:) for update
- Between the 3 last sentences, we change "." -> ";" -> "!", and it changes the logits quite a bit

## 2: Capturing attention outputs

It turns out for path patching, Flash Attention is a road block. In Flash Attention, all the heads are merged together. However, we want to be able to freeze the output and patch each attention head independently.

Test 1: Starts by patching all the attention heads at the last block, one by one

Observation: can confirm the claim from the paper

- Head [11][1] seems to be an Name Mover Head. Knocking it out reduce "Mary" logit.
- Head [11][10] seems to be the Negative Name Mover Head. Knocking it out increases "Mary" logit

Example of knocking out an unrelated head: "Mary" logit stays around 0.4x

```python
--------------------------------------------------------------------------------
Testing head [11][0]
When Mike and Tom went to the store, Rise gave a drink to
0.13  them
0.12  the
0.09  Tom
When Mary and John went to the store, John gave a drink to
0.45  Mary
0.21  them
0.07  the
When Mary and John went to the store, John gave a drink to
0.46  Mary
0.20  them
0.07  the
```

Example of knocking out a Name Mover Head. "Mary" logit suffers.

```python
--------------------------------------------------------------------------------
Testing head [11][1]
When Mike and Tom went to the store, Rise gave a drink to
0.13  them
0.12  the
0.09  Tom
When Mary and John went to the store, John gave a drink to
0.45  Mary
0.21  them
0.07  the
When Mary and John went to the store, John gave a drink to
0.31  them
0.30  Mary
0.10  the
```

Example of knocking out a Negative Name Mover Head. "Mary" logit increases.

```python
--------------------------------------------------------------------------------
Testing head [11][10]
When Mike and Tom went to the store, Rise gave a drink to
0.13  them
0.12  the
0.09  Tom
When Mary and John went to the store, John gave a drink to
0.45  Mary
0.21  them
0.07  the
When Mary and John went to the store, John gave a drink to
0.65  Mary
0.12  them
0.06  John
```

NOTE: the above results were from _one_ set of sample. It's better to curate a dataset of these and sample more oftens. Going to bed since it's late.

## 3: Fix model to matches HF GPT2

Wrote new `model_test.py`, which detected logits differences between `model` and `pretrained_model`. That means we were not as close to GPT2 as we thought. The main root cause is `GELU`. HF used the approximation form, and we didn't.

Fixed the bug and re-run the notebook. The result doesn't change much (mostly the smaller probs that were changed, not the topk)

## 4: Fix case template

While working on the analyzer, I noticed that the starts token for the 2 sentences case above is not correct. It was without " ".

To get a proper 2 sentences case, we should have another word before the name:

```python
Yesterday Mary and John went to the store. John gave a drink to
0.74  Mary
0.06  them
0.04  the
Yesterday Mary and John went to the store; John gave a drink to
0.66  Mary
0.10  them
0.05  John
Yesterday Mary and John went to the store! John gave a drink to
0.55  Mary
0.08  them
0.06  the
Yesterday Mary and John went to the store. Mary gave a drink to
0.43  John
0.11  them
0.10  the
Yesterday Mary and John went to the store; Mary gave a drink to
0.35  John
0.18  them
0.11  Mary
Yesterday Mary and John went to the store! Mary gave a drink to
0.37  John
0.13  the
0.09  them
```

Observation: now the network correctly predicts the next token " John" and " Mary"

## 5: Test with more names

Given the following 3 templates:

```python
    cases = [
        "When {s1} and {s2} went to the store, {s3} gave a drink to",
        "When {s1} and {s2} went to the park, {s3} gave a leaf to",
        "Yesterday {s1} and {s2} went to the store. {s3} gave a drink to",
    ]
```

We run the tests with random (valid) English names, with 3 forms:

- ABC: all 3 names are different
- ABA: only 2 names, the last name is the same with the first name
- ABB: only 2 names, the last name is the same with the second name

Expectation: for all 3 templates

- ABC: no provided name should appear in the top logits
- ABA: predicts B as the top logit
- ABB: predicts A as the top logit

Observation:

- ABC: as expected, the most common logit is " the"
- ABA: for most cases, the correct name A is predicted as the top logit. However, it depends a lot on the names itself. For example, "Davis" and "May" didn't work out.
- ABB: for most cases, the correct name B is predicted. Same above, it depends on the names

This means the result might changes depending on the languages. Right now we're focusing on testing English names only.

## 6: Capture outputs across a large ABC batch before running Path Patching

Specifically: the head outputs are now run on 128 random name samples, then its `mean` is being captured for Path Patching as the baseline.

No new research observation beside the name itself really influences the logits. Not all names are equal.
