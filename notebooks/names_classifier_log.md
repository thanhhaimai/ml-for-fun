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

[BEST] With `dropout=1`. Seems like we are now in the high bias zone.

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
