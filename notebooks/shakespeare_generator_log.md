# Research Log for Shakespeare Generator

## 1: First run

Config:

```
BATCH_SIZE=64, EPOCHS=10, LEARNING_RATE=0.001, PATIENCE=None, MIN_DELTA=None
ParallelBatchLearner(
model=ShakespeareGenerator(
  (embedding): Embedding(66, 66, padding_idx=0)
)
optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
criterion=CrossEntropyLoss())
```

Result:

```
9/10 -- 6.50s  Train loss  2.4529  Eval loss  2.4542  <<
Training completed. Elapsed time: 65.51s
```

Generation:

```
---------- Predict from empty input:
<|pad|>?
Ro ailos e t ite his yod, sie, hthithas withent nd ayor r m; hiornt chr, soiton'TAn,
RD:

O, ainer
<|pad|>He hat wdie I at ce;
Sofflles d t otwer be s treyo belsphat.
K: trstr giridofay
ADigholof bushausoll
```

## 2: Use `mps` device

Eww very slow:

```
0/10 -- 33.88s  Train loss  2.6931  Eval loss  2.4545  <<
```

On CPU:

```
0/10 -- 6.54s  Train loss  2.6952  Eval loss  2.4562  <<
```

Change BATCH_SIZE=1024 on cpu

```
BATCH_SIZE=1024, EPOCHS=10, LEARNING_RATE=0.001, PATIENCE=None, MIN_DELTA=None
9/10 -- 2.05s  Train loss  2.4616  Eval loss  2.4613  <<
Training completed. Elapsed time: 20.14s
```

Use `model.compile()` is slower

```
9/10 -- 2.35s  Train loss  2.4616  Eval loss  2.4613  <<
Training completed. Elapsed time: 26.83s
```

## 3: Use an AttentionHead + PositionalEmbedding

New model:

```
Config(batch_size=1024, sequence_length=8, embedding_size=32, head_size=32, epochs=10, learning_rate=0.001, patience=None, min_delta=None)
ParallelBatchLearner(
model=ShakespeareGenerator(
  (embedding): Embedding(66, 32)
  (positional_embedding): Embedding(8, 32)
  (head): AttentionHead(
    (query): Linear(in_features=32, out_features=32, bias=False)
    (key): Linear(in_features=32, out_features=32, bias=False)
    (value): Linear(in_features=32, out_features=32, bias=False)
  )
  (linear): Linear(in_features=32, out_features=66, bias=True)
)
optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
criterion=CrossEntropyLoss())
```

Result: takes ~5x more time, but loss reduced to 2.30

```
9/10 -- 10.21s  Train loss  2.3030  Eval loss  2.3073  <<
Training completed. Elapsed time: 104.95s
```

## 4: Use MultiHeadAttention

New model:

```
model=ShakespeareGenerator(
  (embedding): Embedding(66, 32)
  (positional_embedding): Embedding(8, 32)
  (heads): MultiHeadAttention(
    (heads): ModuleList(
      (0-3): 4 x AttentionHead(
        (query): Linear(in_features=32, out_features=8, bias=False)
        (key): Linear(in_features=32, out_features=8, bias=False)
        (value): Linear(in_features=32, out_features=8, bias=False)
      )
    )
  )
  (linear): Linear(in_features=32, out_features=66, bias=True)
)
```

Result: 3x training time, but loss reduced to 2.07

```
9/10 -- 33.31s  Train loss  2.0751  Eval loss  2.0769  <<
Training completed. Elapsed time: 332.04s
```

## 5: Use FeedForward

New model:

```
model=ShakespeareGenerator(
  (embedding): Embedding(66, 32)
  (positional_embedding): Embedding(8, 32)
  (heads): MultiHeadAttention(
    (heads): ModuleList(
      (0-3): 4 x AttentionHead(
        (query): Linear(in_features=32, out_features=8, bias=False)
        (key): Linear(in_features=32, out_features=8, bias=False)
        (value): Linear(in_features=32, out_features=8, bias=False)
      )
    )
  )
  (feed_forward): FeedForward(
    (linear): Linear(in_features=32, out_features=32, bias=True)
    (gelu): GELU(approximate='none')
  )
  (linear): Linear(in_features=32, out_features=66, bias=True)
)
```

Result: small improvement with loss down to 2.03

```
9/10 -- 34.27s  Train loss  2.0331  Eval loss  2.0320  <<
Training completed. Elapsed time: 348.24s
```

## 6: Add Residual Pathway

New model:

```
model=ShakespeareGenerator(
  (embedding): Embedding(66, 32)
  (positional_embedding): Embedding(8, 32)
  (blocks): Sequential(
    (0): Block(
      (heads): MultiHeadAttention(
        (heads): ModuleList(
          (0-3): 4 x AttentionHead(
            (query): Linear(in_features=32, out_features=8, bias=False)
            (key): Linear(in_features=32, out_features=8, bias=False)
            (value): Linear(in_features=32, out_features=8, bias=False)
          )
        )
        (projection): Linear(in_features=32, out_features=32, bias=True)
      )
      (feed_forward): FeedForward(
        (linear): Linear(in_features=32, out_features=32, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=32, out_features=32, bias=True)
      )
    )
  )
  (linear): Linear(in_features=32, out_features=66, bias=True)
)
```

Result: down to 2.0 loss!

```
9/10 -- 36.53s  Train loss  2.0126  Eval loss  2.0098  <<
Training completed. Elapsed time: 362.16s
```

## 7: Add LayerNorm

New model:

```
model=ShakespeareGenerator(
  (embedding): Embedding(66, 32)
  (positional_embedding): Embedding(8, 32)
  (blocks): Sequential(
    (0): Block(
      (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (heads): MultiHeadAttention(
        (heads): ModuleList(
          (0-3): 4 x AttentionHead(
            (query): Linear(in_features=32, out_features=8, bias=False)
            (key): Linear(in_features=32, out_features=8, bias=False)
            (value): Linear(in_features=32, out_features=8, bias=False)
          )
        )
        (projection): Linear(in_features=32, out_features=32, bias=True)
      )
      (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (linear): Linear(in_features=32, out_features=32, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=32, out_features=32, bias=True)
      )
    )
    (1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  )
  (linear): Linear(in_features=32, out_features=66, bias=True)
)
```

Result:

```
9/10 -- 37.11s  Train loss  1.9709  Eval loss  1.9675  <<
Training completed. Elapsed time: 377.62s
```

## 8: Add Dropout

New model:

```
model=ShakespeareGenerator(
  (embedding): Embedding(66, 32)
  (positional_embedding): Embedding(8, 32)
  (blocks): Sequential(
    (0): Block(
      (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (heads): MultiHeadAttention(
        (heads): ModuleList(
          (0-3): 4 x AttentionHead(
            (query): Linear(in_features=32, out_features=8, bias=False)
            (key): Linear(in_features=32, out_features=8, bias=False)
            (value): Linear(in_features=32, out_features=8, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (projection): Linear(in_features=32, out_features=32, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (linear): Linear(in_features=32, out_features=32, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=32, out_features=32, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  )
  (linear): Linear(in_features=32, out_features=66, bias=True)
)
```

Result:

```
9/10 -- 33.22s  Train loss  2.0727  Eval loss  2.0079  <<
Training completed. Elapsed time: 328.86s
```

## 9: Scale up

New config:

```
Config(batch_size=1024, sequence_length=64, embedding_size=64, num_heads=8, epochs=100, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001)
```

Result: may not finish hahaha. Runs very slow, but it learns so well!

```
0/100 -- 352.04s  Train loss  2.4108  Eval loss  2.0206  <<
1/100 -- 354.38s  Train loss  1.9703  Eval loss  1.8294  <<
2/100 -- 355.16s  Train loss  1.8667  Eval loss  1.7627  <<
```

## 10: Switch to Desktop

1080 Ti

```
Config(batch_size=1024, sequence_length=8, embedding_size=32, num_heads=4, epochs=100, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001)
model=ShakespeareGenerator(
  (embedding): Embedding(66, 32)
  (positional_embedding): Embedding(8, 32)
  (blocks): Sequential(
    (0): Block(
      (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (heads): MultiHeadAttention(
        (heads): ModuleList(
          (0-3): 4 x AttentionHead(
            (query): Linear(in_features=32, out_features=8, bias=False)
            (key): Linear(in_features=32, out_features=8, bias=False)
            (value): Linear(in_features=32, out_features=8, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (projection): Linear(in_features=32, out_features=32, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (linear): Linear(in_features=32, out_features=32, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=32, out_features=32, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  )
  (linear): Linear(in_features=32, out_features=66, bias=True)
)
```

Result: Nice, faster!

```
0/100 -- 4.43s  Train loss  2.5861  Eval loss  2.2469  <<
```

NOTE: I'm using a poor 1080 Ti, which has lower FP16 performance than FP32. So `torch.autocast` wouldn't help here.

## 11: Keep dataset on CPU, and only move the batch to CUDA

Also with that, scale up the batch until OOM. The dataloader has 4 worker, and `pin_memory=True`

```python
Config(batch_size=8192, sequence_length=64, embedding_size=64, num_heads=8, num_blocks=2, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))
ParallelBatchLearner
model=ShakespeareGenerator(
  (embedding): Embedding(66, 64)
  (positional_embedding): Embedding(64, 64)
  (blocks): Sequential(
    (0): Block(
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (heads): MultiHeadAttention(
        (heads): ModuleList(
          (0-7): 8 x AttentionHead(
            (query): Linear(in_features=64, out_features=8, bias=False)
            (key): Linear(in_features=64, out_features=8, bias=False)
            (value): Linear(in_features=64, out_features=8, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (projection): Linear(in_features=64, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (linear): Linear(in_features=64, out_features=256, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=256, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (1): Block(
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (heads): MultiHeadAttention(
        (heads): ModuleList(
          (0-7): 8 x AttentionHead(
            (query): Linear(in_features=64, out_features=8, bias=False)
            (key): Linear(in_features=64, out_features=8, bias=False)
            (value): Linear(in_features=64, out_features=8, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (projection): Linear(in_features=64, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (linear): Linear(in_features=64, out_features=256, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=256, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  )
  (linear): Linear(in_features=64, out_features=66, bias=True)
)
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
ShakespeareGenerator                          --
├─Embedding: 1-1                              4,224
├─Embedding: 1-2                              4,096
├─Sequential: 1-3                             --
│    └─Block: 2-1                             --
│    │    └─LayerNorm: 3-1                    128
│    │    └─MultiHeadAttention: 3-2           16,448
│    │    └─LayerNorm: 3-3                    128
│    │    └─FeedForward: 3-4                  33,088
│    └─Block: 2-2                             --
│    │    └─LayerNorm: 3-5                    128
│    │    └─MultiHeadAttention: 3-6           16,448
│    │    └─LayerNorm: 3-7                    128
│    │    └─FeedForward: 3-8                  33,088
│    └─LayerNorm: 2-3                         128
├─Linear: 1-4                                 4,290
======================================================================
Total params: 112,322
Trainable params: 112,322
Non-trainable params: 0
======================================================================
optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
criterion=CrossEntropyLoss()
```

Result:

```python
0/1 -- 90.67s  Train loss  2.9852  Eval loss  2.5462  <<
Training completed. Elapsed time: 94.32s
```

Fix `pin_memory` wasn't being active for custom batch

```python
0/1 -- 89.90s  Train loss  2.9852  Eval loss  2.5462  <<
Training completed. Elapsed time: 93.49s
```

A longer train with reduced FFN expansion faction 4 to 2

```python
Config(batch_size=8192, sequence_length=64, embedding_size=64, num_heads=8, num_blocks=2, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))

      (feed_forward): FeedForward(
        (linear): Linear(in_features=64, out_features=128, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=128, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
```

Trained for a bit more than 2h on 1080 Ti:

```python
99/100 -- 84.39s  Train loss  1.5372  Eval loss  1.4405  
Training completed. Elapsed time: 8384.28s
```

Output:

```text
---------- Predict from empty input:

MENENIUS:
What you for thee: and I long help, I, give;
I will not beared,
My gaon, sats Edward have is the brutht
Whom Cortury may aslain. The comport appears,
Shall prushione got with devish, embrge; 'ad: will and
but so him him mouned. What, my lord,
That see telp him scries maless and hereop.

HERMIOR:
O, ported you, sir, of such my name sprifters;
But not did basts to ubeg what in to married;
And we know't the-piterly.

LADY CAMILLO:
Unow thy speak a care, that negin.
God Spiright to have justice of Vinences
he which but nerce thee what your hight.

First.

Second MARCIUS:
Tit you make me;
Puts propers are you high of thine other,
As welt strange usurlige anotheron'd death people you
Which see-for peceedly with him
May be patient to do say in the suppersman.
And you highness! I would long day!
Thee you that I ounnot as you quarvil in a tis much
I officious, of like to come, Hencer.
Your father's life, down, O, I will him'd I
Lave three brought to cannot have made.
She mine for this
```

## 12: Let's scale it up more

```python
Config(batch_size=1024, sequence_length=128, embedding_size=128, num_heads=8, num_blocks=4, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))
```

Woa, the result is decent after just _one_ epoch:

```
0/1 -- 487.05s  Train loss  1.9666  Eval loss  1.4900  <<
Training completed. Elapsed time: 505.24s
```

```
---------- Predict from empty input:

That'st go have behing nothine to shalt speak?

LENSIUS:
O then nearls from is she just and underso
And, for the's not the saw it; teep in the
were methouch such fearfulliked and I but be and shall
March of Son though that alter saler to the never abos--
What brother night did and fastsherls childress?
O, and I day my lord by to great thy honour:
Aland I do thou for-th bosom soshorts lebeguest
A pity, for relm now. He'lt.

NORTHASUMNE:
Stanly unbague: this thinkle, and buttle give me
have ened for Gentlemont, I woe the it-dail;
And if where pray you the speak news, the divinker
And for in self, if I compart and the buirt,
Shall in it my barish, fasting quien, and their
Too kill bear their burnisse are for love,
And you the soul pretty of the vatual,
Thouy hast-kingdown by his quay sir.
```

After 20 epochs: still train very well without sign of overfitting.

```
19/20 -- 586.67s  Train loss  1.2016  Eval loss  1.0941  <<
Training completed. Elapsed time: 10998.25s
```

And the text feels more English, but still not GPT level (expected)

```
---------- Predict from empty input:

PRINCE EDWARD:
Upon him, every gracious leaves me:
Poor Katch, let us faith. Doth this hung'd?

KING RICHARD II:
Then night he may come with death?

BENVOLIO:
What, ho, were standing.
Your nature fellows with mad?

ROMEO:
Are the noise of
An e'er hail Thursday! O, in vain All-Seein, why, what say
I'll acquaint his house goes to me to grant
What is cursed of holy love,
And cousin by his gage
Below in war: I offend him, but wanting.

VINCENTIO:
O, the heavens! I'll cousin of kind itself?

BISHOP OF CARLISLE:
O shall! the news of deeds, I'll seven with her!
My meerness could not know not my natural king:
The Roman shall confess the morns are spent alus.
Then tidings and leave to keep her name:
Farewell: then, by your signhyer'd hand,
To youthful, that Henry lords of Marcius,
Who thus chequent'st you out, and labour out on their
Duisher, dishonour!

First Murderer:
Who wilt thou hast thou swear'st?

PRINCE EDWARD:
Trust! gavest you fail.
```

For fun, run a high sequence:

```python
Config(batch_size=64, sequence_length=512, embedding_size=128, num_heads=8, num_blocks=4, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))

0/1 -- 4586.11s  Train loss  1.4903  Eval loss  1.1940  <<
Training completed. Elapsed time: 4762.39s
```

Lower `embedding_size=32`

```python
Config(batch_size=64, sequence_length=512, embedding_size=32, num_heads=8, num_blocks=4, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))

0/1 -- 3987.07s  Train loss  2.2541  Eval loss  1.8976  <<
Training completed. Elapsed time: 4140.31s
```

```python
Config(batch_size=64, sequence_length=512, embedding_size=256, num_heads=8, num_blocks=4, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))

0/1 -- 5621.24s  Train loss  1.2046  Eval loss  0.8486  <<
Training completed. Elapsed time: 5892.93s
```

It scales really well:

```python
Config(batch_size=64, sequence_length=512, embedding_size=256, num_heads=8, num_blocks=4, epochs=5, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))

4/5 -- 5263.90s  Train loss  0.7821  Eval loss  0.4778  <<
Training completed. Elapsed time: 26561.78s
```

```
---------- Predict from empty input:

ISABELLA:
So show you joy! You be the way in grace?

ISABELLA:
Back most bitterly, bonds, but having the luke.

DUKE VINCENTIO:
Under thee not.

ISABELLA:
Good morrow, as it were not, but stisens not a doing,
I would be revenged on thee to thy admiral:
The antigence of thy joy
Were join'd with blissing hath earth as such
As he would proclaim'd it from meaner.

DUKE VINCENTIO:
Stanley, my children,
If I do not know thee since that I have his,
His making sensible nor woe for your right:
But since the common exactly dove
For his powers. But in this young Mariana
Those comfort, sometimes yours, my wife's due is
And tell the envy of his instruments,
If ever he should forget forget the alter's intent,
Why, he thoughts his life upon his state,
And, made a frank from his royal prince,
And make him append to this bigger flocks.
Come, come, good Clarence; whom fast thou behind
Is famous golden? keep me with me to-day.
```

## 13: Flash Attention

```python
Config(batch_size=8192, sequence_length=64, embedding_size=64, num_heads=8, num_blocks=2, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))
ParallelBatchLearner
model=ShakespeareGenerator(
  (embedding): Embedding(66, 64)
  (positional_embedding): Embedding(64, 64)
  (blocks): Sequential(
    (0): Block(
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (heads): MultiHeadAttention(
        (qkv_fc): Linear(in_features=64, out_features=192, bias=True)
        (projection): Linear(in_features=64, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (linear): Linear(in_features=64, out_features=128, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=128, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (1): Block(
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (heads): MultiHeadAttention(
        (qkv_fc): Linear(in_features=64, out_features=192, bias=True)
        (projection): Linear(in_features=64, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (linear): Linear(in_features=64, out_features=128, bias=True)
        (gelu): GELU(approximate='none')
        (projection): Linear(in_features=128, out_features=64, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  )
  (linear): Linear(in_features=64, out_features=66, bias=True)
)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
ShakespeareGenerator                     --
├─Embedding: 1-1                         4,224
├─Embedding: 1-2                         4,096
├─Sequential: 1-3                        --
│    └─Block: 2-1                        --
│    │    └─LayerNorm: 3-1               128
│    │    └─MultiHeadAttention: 3-2      16,640
│    │    └─LayerNorm: 3-3               128
│    │    └─FeedForward: 3-4             16,576
│    └─Block: 2-2                        --
│    │    └─LayerNorm: 3-5               128
│    │    └─MultiHeadAttention: 3-6      16,640
│    │    └─LayerNorm: 3-7               128
│    │    └─FeedForward: 3-8             16,576
│    └─LayerNorm: 2-3                    128
├─Linear: 1-4                            4,290
=================================================================
Total params: 79,682
Trainable params: 79,682
Non-trainable params: 0
```

Result:

```python
0/1 -- 42.31s  Train loss  3.0843  Eval loss  2.6122  <<
Training completed. Elapsed time: 44.46s
```

```
Config(batch_size=256, sequence_length=512, embedding_size=256, num_heads=16, num_blocks=4, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))

0/1 -- 3337.12s 	Train loss 	1.4413 	Eval loss 	1.0959 	<<
Training completed. Elapsed time: 3439.49s
```

```
Config(batch_size=256, sequence_length=512, embedding_size=256, num_heads=8, num_blocks=4, epochs=1, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001, device=device(type='cuda'))
0/1 -- 3022.46s 	Train loss 	1.4066 	Eval loss 	1.0586 	<<
Training completed. Elapsed time: 3107.95s
```