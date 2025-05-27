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
9/10 -- 6.50s 	Train loss 	2.4529 	Eval loss 	2.4542 	<<
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
0/10 -- 33.88s 	Train loss 	2.6931 	Eval loss 	2.4545 	<<
```

On CPU:
```
0/10 -- 6.54s 	Train loss 	2.6952 	Eval loss 	2.4562 	<<
```

Change BATCH_SIZE=1024 on cpu
```
BATCH_SIZE=1024, EPOCHS=10, LEARNING_RATE=0.001, PATIENCE=None, MIN_DELTA=None
9/10 -- 2.05s 	Train loss 	2.4616 	Eval loss 	2.4613 	<<
Training completed. Elapsed time: 20.14s
```

Use `model.compile()` is slower
```
9/10 -- 2.35s 	Train loss 	2.4616 	Eval loss 	2.4613 	<<
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
9/10 -- 10.21s 	Train loss 	2.3030 	Eval loss 	2.3073 	<<
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
9/10 -- 33.31s 	Train loss 	2.0751 	Eval loss 	2.0769 	<<
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
9/10 -- 34.27s 	Train loss 	2.0331 	Eval loss 	2.0320 	<<
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
9/10 -- 36.53s 	Train loss 	2.0126 	Eval loss 	2.0098 	<<
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
9/10 -- 37.11s 	Train loss 	1.9709 	Eval loss 	1.9675 	<<
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
9/10 -- 33.22s 	Train loss 	2.0727 	Eval loss 	2.0079 	<<
Training completed. Elapsed time: 328.86s
```

## 9: Scale up

New config:
```
Config(batch_size=1024, sequence_length=64, embedding_size=64, num_heads=8, epochs=100, dropout=0.1, learning_rate=0.001, patience=30, min_delta=0.001)
```

Result: may not finish hahaha. Runs very slow, but it learns so well!
```
0/100 -- 352.04s 	Train loss 	2.4108 	Eval loss 	2.0206 	<<
1/100 -- 354.38s 	Train loss 	1.9703 	Eval loss 	1.8294 	<<
2/100 -- 355.16s 	Train loss 	1.8667 	Eval loss 	1.7627 	<<
```
