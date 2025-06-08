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

- Maybe I didn't fully reproduce GPT2 correctly? -- not the case
  - Confirmed that the HF pretrained model output matched my model output
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

- This feels like the original research result might be more fragile than the paper implies
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
