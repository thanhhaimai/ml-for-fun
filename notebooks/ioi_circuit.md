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

## 7: Metrics time

Implemented:

- KL Divergence
- Jensen-Shannon Divergence
- Total Variation Distance
- L2 Distance

Observation (for the last block)

- Head 10 is Negative Name Mover head
- Head 1 and Head 2 both contributes largely to the "Mary" output

```python
When Mary and John went to the store, John gave a drink to

================================================================================
SUMMARY: Most impactful heads (by KL divergence)
================================================================================
#1  Head [11][10]  KL: 0.1395 TV: 0.2426 L2: 0.2691 Mary prob: -0.2424 John prob: 0.0054 Logit diff: -0.5275 
#2  Head [11][1]  KL: 0.0753 TV: 0.1808 L2: 0.1767 Mary prob: 0.1469 John prob: 0.0324 Logit diff: -0.3763 
#3  Head [11][2]  KL: 0.0360 TV: 0.1259 L2: 0.1393 Mary prob: 0.1255 John prob: -0.0022 Logit diff: 0.3666 
#4  Head [11][3]  KL: 0.0185 TV: 0.0921 L2: 0.0999 Mary prob: 0.0895 John prob: -0.0087 Logit diff: 0.3589 
#5  Head [11][6]  KL: 0.0062 TV: 0.0533 L2: 0.0588 Mary prob: 0.0530 John prob: -0.0028 Logit diff: 0.1713 
#6  Head [11][8]  KL: 0.0046 TV: 0.0326 L2: 0.0243 Mary prob: -0.0149 John prob: 0.0057 Logit diff: -0.1336 
#7  Head [11][11]  KL: 0.0032 TV: 0.0388 L2: 0.0398 Mary prob: 0.0348 John prob: 0.0033 Logit diff: 0.0242 
#8  Head [11][0]  KL: 0.0008 TV: 0.0160 L2: 0.0122 Mary prob: -0.0084 John prob: -0.0030 Logit diff: 0.0299 
#9  Head [11][9]  KL: 0.0006 TV: 0.0164 L2: 0.0171 Mary prob: 0.0154 John prob: -0.0001 Logit diff: 0.0368 
#10  Head [11][7]  KL: 0.0005 TV: 0.0144 L2: 0.0151 Mary prob: -0.0138 John prob: 0.0001 Logit diff: -0.0329 
#11  Head [11][5]  KL: 0.0002 TV: 0.0067 L2: 0.0045 Mary prob: -0.0025 John prob: 0.0027 Logit diff: -0.0516 
#12  Head [11][4]  KL: 0.0001 TV: 0.0049 L2: 0.0037 Mary prob: -0.0034 John prob: 0.0002 Logit diff: -0.0114
```
