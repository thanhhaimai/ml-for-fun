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
- Maybe the GPT2 weights from HF is not 100% the same with the TensorFlow GPT2 version?

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
- Between the 3 last sentences, we change "." -> ";", and it changes the logits quite a bit
