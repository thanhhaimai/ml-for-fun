# Research Log for Names Classifier

## 1: RNN first run

Config: `seed=42`

```python
Config(batch_size=1024, learning_rate=0.001, epochs=500, patience=30, min_delta=0.0001, device=device(type='cpu'), vocab_size=58, class_size=18, hidden_size=64, num_layers=1, bidirectional=False, activation='tanh', dropout=0.1)
ParallelBatchLearner
model=NamesClassifierRNN(
  (rnn): RNN(58, 64)
  (fc): Linear(in_features=64, out_features=18, bias=True)
)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
NamesClassifierRNN                       --
├─RNN: 1-1                               7,936
├─Linear: 1-2                            1,170
=================================================================
Total params: 9,106
Trainable params: 9,106
Non-trainable params: 0
=================================================================
optimizer=AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.01
)
criterion=CrossEntropyLoss()
```

Result:

```python
146/500 -- 0.09s  train loss  0.7681  eval loss  0.8330  
Early stopping at epoch 146 best_eval_loss=0.7880
Training completed. Elapsed time: 14.44s
```

Change hidden_size=128. Similar performance.

```python
100/500 -- 0.28s  Train loss  0.5955  Eval loss  0.8350  
Early stopping at epoch 100 best_eval_loss=0.7806
Training completed. Elapsed time: 24.14s
```

Back to hidden_size=64. Change to `relu`. Better performance.

```python
215/500 -- 0.09s  Train loss  0.7679  Eval loss  0.8129  
Early stopping at epoch 215 best_eval_loss=0.7689
Training completed. Elapsed time: 20.46s
```

Use `bidirectional=True`. It seems to not train as well.

```python
177/500 -- 0.15s  Train loss  0.9527  Eval loss  0.8093  
Early stopping at epoch 177 best_eval_loss=0.7796
Training completed. Elapsed time: 30.44s
```

Back to `bidrectional=False`. Increase `num_layer=2`. Better performance.

```python
135/500 -- 0.14s  Train loss  0.7369  Eval loss  0.7618  
Early stopping at epoch 135 best_eval_loss=0.7285
Training completed. Elapsed time: 18.82s
```

Increase to `num_layers=3`. Not good.

```python
105/500 -- 0.18s  Train loss  0.7093  Eval loss  0.8370  
Early stopping at epoch 105 best_eval_loss=0.7966
Training completed. Elapsed time: 22.00s
```

Try `bidirection=True` with `num_layers=2`. Not good.

```python
98/500 -- 0.49s  Train loss  0.6881  Eval loss  0.8137  
Early stopping at epoch 98 best_eval_loss=0.7496
Training completed. Elapsed time: 30.95s
```

BUG: found a bug, `dropout` was not effective. All runs above were with `dropout=0`

With `dropout=1`. Seems like we are now in the high bias zone.

```python
114/500 -- 0.18s  Train loss  0.8196  Eval loss  0.7435  
Early stopping at epoch 114 best_eval_loss=0.7126
Training completed. Elapsed time: 30.65s
```

Increase `hidden_size=128`. Overfitting, not good

```python
78/500 -- 0.45s  Train loss  0.6009  Eval loss  0.8686  
Early stopping at epoch 78 best_eval_loss=0.8363
Training completed. Elapsed time: 37.80s
```

## 2: Now running on `cuda`

```python
Config(batch_size=1024, learning_rate=0.001, epochs=500, patience=30, min_delta=0.0001, device=device(type='cuda'), vocab_size=58, class_size=18, hidden_size=64, num_layers=2, bidirectional=False, activation='relu', dropout=0.2)
ParallelBatchLearner
model=NamesClassifierRNN(
  (rnn): RNN(58, 64, num_layers=2)
  (fc): Linear(in_features=64, out_features=18, bias=True)
)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
NamesClassifierRNN                       --
├─RNN: 1-1                               16,256
├─Linear: 1-2                            1,170
=================================================================
Total params: 17,426
Trainable params: 17,426
Non-trainable params: 0
=================================================================
optimizer=AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.01
)
criterion=CrossEntropyLoss()
```

Result: faster, but also different loss!

```python
137/500 -- 0.10s  Train loss  0.6466  Eval loss  0.9010  
Early stopping at epoch 137 best_eval_loss=0.7999
Training completed. Elapsed time: 13.32s
```

Here is the `cpu` run for comparision:

```python
135/500 -- 0.14s  Train loss  0.7369  Eval loss  0.7618  
Early stopping at epoch 135 best_eval_loss=0.7285
Training completed. Elapsed time: 20.02s
```

Trying `cuda` with a different seed=52

```python
83/500 -- 0.09s  Train loss  0.9001  Eval loss  0.8573  
Early stopping at epoch 83 best_eval_loss=0.8350
Training completed. Elapsed time: 8.01s
```

BUG: `bidirectional` wasn't working. Fixed and use `bidrectional=True` on `cuda` with `seed=42`

```python
88/500 -- 0.12s  Train loss  0.5794  Eval loss  0.8200  
Early stopping at epoch 88 best_eval_loss=0.7680
Training completed. Elapsed time: 11.04s
```

## 3: Use LSTM

```python
Config(batch_size=1024, learning_rate=0.001, epochs=500, patience=30, min_delta=0.0001, device=device(type='cuda'), vocab_size=58, class_size=18, hidden_size=64, num_layers=2, bidirectional=True, activation='relu', dropout=0.2)
ParallelBatchLearner
model=NamesClassifierLSTM(
  (lstm): LSTM(58, 64, num_layers=2, dropout=0.2, bidirectional=True)
  (fc): Linear(in_features=128, out_features=18, bias=True)
)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
NamesClassifierLSTM                      --
├─LSTM: 1-1                              162,816
├─Linear: 1-2                            2,322
=================================================================
Total params: 165,138
Trainable params: 165,138
Non-trainable params: 0
=================================================================
optimizer=AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.01
)
criterion=CrossEntropyLoss()
```

Result:

```python
114/500 -- 0.16s  Train loss  0.5309  Eval loss  0.8451  
Early stopping at epoch 114 best_eval_loss=0.7948
Training completed. Elapsed time: 18.47s
```

For comparision, this is the RNN run:

```python
Config(batch_size=1024, learning_rate=0.001, epochs=500, patience=30, min_delta=0.0001, device=device(type='cuda'), vocab_size=58, class_size=18, hidden_size=64, num_layers=2, bidirectional=True, activation='relu', dropout=0.2)
ParallelBatchLearner
model=NamesClassifierRNN(
  (rnn): RNN(58, 64, num_layers=2, dropout=0.2, bidirectional=True)
  (fc): Linear(in_features=128, out_features=18, bias=True)
)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
NamesClassifierRNN                       --
├─RNN: 1-1                               40,704
├─Linear: 1-2                            2,322
=================================================================
Total params: 43,026
Trainable params: 43,026
Non-trainable params: 0
=================================================================
optimizer=AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.01
)
criterion=CrossEntropyLoss()
```

Result:

```python
91/500 -- 0.13s  Train loss  0.6071  Eval loss  0.7991  
Early stopping at epoch 91 best_eval_loss=0.7608
Training completed. Elapsed time: 11.75s
```

## 4: Use GRU

```python
Config(batch_size=1024, learning_rate=0.001, epochs=500, patience=30, min_delta=0.0001, device=device(type='cuda'), vocab_size=58, class_size=18, hidden_size=64, num_layers=2, bidirectional=True, activation='relu', dropout=0.2)
ParallelBatchLearner
model=NamesClassifierGRU(
  (gru): GRU(58, 64, num_layers=2, dropout=0.2, bidirectional=True)
  (fc): Linear(in_features=128, out_features=18, bias=True)
)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
NamesClassifierGRU                       --
├─GRU: 1-1                               122,112
├─Linear: 1-2                            2,322
=================================================================
Total params: 124,434
Trainable params: 124,434
Non-trainable params: 0
=================================================================
optimizer=AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.01
)
criterion=CrossEntropyLoss()
```

Result: Takes more time to run, but better result.

```python
115/500 -- 0.28s  Train loss  0.3661  Eval loss  0.7808  
Early stopping at epoch 115 best_eval_loss=0.7187
Training completed. Elapsed time: 32.82s
```

## 5: Use Embedding with simple RNN

```python
Config(batch_size=1024, learning_rate=0.001, epochs=500, patience=30, min_delta=0.0001, device=device(type='cuda'), vocab_size=58, class_size=18, embedding_size=32, hidden_size=64, num_layers=2, bidirectional=True, activation='relu', dropout=0.2)
ParallelBatchLearner
model=NamesClassifierRNN(
  (embedding): Embedding(58, 32, padding_idx=0)
  (ln1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  (rnn): RNN(32, 64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
  (ln2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (fc): Linear(in_features=128, out_features=18, bias=True)
)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
NamesClassifierRNN                       --
├─Embedding: 1-1                         1,856
├─LayerNorm: 1-2                         64
├─RNN: 1-3                               37,376
├─LayerNorm: 1-4                         256
├─Linear: 1-5                            2,322
=================================================================
Total params: 41,874
Trainable params: 41,874
Non-trainable params: 0
=================================================================
optimizer=AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.01
)
criterion=CrossEntropyLoss()
```

Result:

```python
79/500 -- 0.13s  Train loss  0.4881  Eval loss  0.8062  
Early stopping at epoch 79 best_eval_loss=0.7303
Training completed. Elapsed time: 10.52s
```

Use LSTM

```python
73/500 -- 0.19s  Train loss  0.2278  Eval loss  0.8273  
Early stopping at epoch 73 best_eval_loss=0.6913
Training completed. Elapsed time: 14.46s
```

[BEST] Use GRU

```python
75/500 -- 0.19s  Train loss  0.2721  Eval loss  0.7809  
Early stopping at epoch 75 best_eval_loss=0.6728
Training completed. Elapsed time: 14.29s
```

Reduce `hidden_size=32`

```python
94/500 -- 0.14s  Train loss  0.4863  Eval loss  0.7247  
Early stopping at epoch 94 best_eval_loss=0.6862
Training completed. Elapsed time: 13.10s
```
