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