# Research Log for Names Classifier

## 1: First run

Config:

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

```
146/500 -- 0.09s  train loss  0.7681  eval loss  0.8330  
Early stopping at epoch 146 best_eval_loss=0.7880
Training completed. Elapsed time: 14.44s
```
