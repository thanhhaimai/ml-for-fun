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
