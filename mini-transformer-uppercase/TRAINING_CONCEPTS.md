# Training Concepts: Dataset Size, Epochs, Batch Size, and Weight Updates

A beginner-friendly guide for the `mini-transformer-uppercase` PyTorch project.

## 1. What is an epoch?

An **epoch** is one complete pass through the training dataset.

If the dataset contains **12,000 examples**:

- **1 epoch:** the model processes 12,000 examples.
- **30 epochs:** the model processes 12,000 examples 30 times.
- **Total examples processed:** `12,000 × 30 = 360,000`.

This does **not** mean 360,000 unique examples. The same dataset is revisited, usually in a shuffled order.

## 2. What is batch size?

**Batch size** is the number of examples processed together to compute a training update.

With **12,000 examples** and **batch size 64**:

- `12,000 ÷ 64 = 187.5`
- Round up: **188 batches per epoch**.
- There are 187 full batches of 64 examples and one final batch of 32.

This assumes the data loader does not discard the incomplete final batch (`drop_last=False`).

## 3. What is a weight?

A neural network contains trainable numerical parameters called **weights** (and often biases). These parameters influence the model's predictions. They usually start with randomized values and are adjusted during training.

For the uppercase Transformer, the model learns to predict the target characters from input examples such as:

```text
Input:  hello
Target: HELLO
```

## 4. What is one weight update?

One **weight update** (also called an **optimizer step**) is when the optimizer adjusts the model's trainable parameters using gradients computed from a batch.

A typical PyTorch training step:

```python
optimizer.zero_grad()                 # Clear old gradients
predictions = model(inputs)          # Forward pass
loss = criterion(predictions, targets) # Calculate prediction error
loss.backward()                      # Compute gradients
optimizer.step()                     # Update model parameters ONCE
```

Each call to `optimizer.step()` counts as **one optimizer update**. One update can change many individual parameters at the same time.

### How a batch is learned

1. **Forward pass:** predict output for the batch.
2. **Loss:** compare predictions with the correct targets.
3. **Backpropagation:** compute gradients of the loss with respect to parameters.
4. **Optimizer step:** adjust parameters to try to reduce future loss.

## 5. How do we get 5,640 updates?

Given:

| Setting | Value |
|---|---:|
| Dataset size | 12,000 |
| Batch size | 64 |
| Epochs | 30 |

**Batches per epoch**:

```text
ceil(12,000 / 64) = 188
```

**Total optimizer updates**:

```text
188 batches/epoch × 30 epochs = 5,640 updates
```

**Total examples processed**:

```text
12,000 examples/epoch × 30 epochs = 360,000 examples
```

These formulas assume one optimizer step per batch, no gradient accumulation, and no discarded last batch.

## 6. What is learning rate?

The **learning rate** influences how much the model's weights change on an update.

For simple gradient descent:

```text
new_weight = old_weight - learning_rate × gradient
```

Example:

```text
old_weight    = 0.500
learning_rate = 0.010
gradient      = 0.200
new_weight    = 0.500 - (0.010 × 0.200) = 0.498
```

The project may use **AdamW**, whose actual update formula is more sophisticated than basic gradient descent.

## 7. How do these settings affect training?

| Setting | Increasing it usually means | Caution |
|---|---|---|
| Training dataset size | More examples and more updates per epoch at the same batch size | Longer epochs; more examples alone do not guarantee better generalization |
| Epochs | More passes through the dataset | Can overfit |
| Batch size | More examples per update, fewer updates per epoch | Higher memory usage; accuracy not guaranteed to improve |
| Learning rate | Larger parameter adjustments | Too high can destabilize training |

**Validation data** helps detect when the model stops improving on unseen examples.

## 8. Example commands

```bash
# Generate dataset (if not already generated)
python dataset.py

# Train for 30 epochs with batches of 64
python train.py --epochs 30 --batch-size 64

# Alternative experiment: more epochs, smaller batches
python train.py --epochs 50 --batch-size 32

# Test the trained model
python inference.py --text "hello danish how are you"
```

The second training command is an **experiment**, not a guarantee of better results. Compare validation loss and exact-match accuracy.

## 9. Why might unfamiliar text produce incorrect uppercase output?

For example:

```text
Input:  hello danish how are you weqeqzaxsc
Model:  HELLO DANISH HOW ARE YOU WEXAZY
Target: HELLO DANISH HOW ARE YOU WEQEQZAXSC
```

The model may have learned familiar character patterns without learning a reliable character-by-character transformation. To improve generalization:

- Include random strings and mixed-case inputs in training.
- Hold out **entire examples** for validation and testing.
- Track **exact-match accuracy** (entire predicted string correct), not only token accuracy.
- Check that all input characters are supported by the tokenizer.
- Adjust architecture and training only after inspecting the metrics.

## 10. How do I increase the dataset size?

`dataset.py` controls dataset generation. Its default sizes are:

| Split | Examples | Purpose |
|---|---:|---|
| Training | 12,000 | Used to update the model's weights |
| Validation | 1,200 | Used to compare performance after each epoch and select the best checkpoint |
| Test | 1,200 | Reserved for a separate evaluation after training |

You can change these sizes from the command line without editing the code. Run these commands from the `mini-transformer-uppercase` directory:

```bash
python dataset.py --train-size 50000 --val-size 5000 --test-size 5000
```

**This overwrites** `data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl`; it does not append examples. The generator creates unique input sentences and prevents overlap between the three splits within that generation run.

To preserve the original dataset, generate a larger one in a separate directory and tell training to use it:

```bash
python dataset.py --output data-large --train-size 50000 --val-size 5000 --test-size 5000
python train.py --data data-large --epochs 30 --batch-size 64 --checkpoint checkpoints/large-best.pt
```

The separate checkpoint path also preserves the original `checkpoints/best.pt`. Each invocation of `train.py` starts a new model from scratch; generating more data does not automatically continue training an existing checkpoint.

### How does a larger dataset affect training?

With **50,000 training examples**, **batch size 64**, and **30 epochs**:

```text
Batches per epoch     = ceil(50,000 / 64) = 782
Total optimizer steps = 782 × 30 = 23,460
Total examples seen   = 50,000 × 30 = 1,500,000
```

Increasing validation or test size does not add optimizer updates. Validation adds evaluation work each epoch; the current training script does not evaluate the test split automatically.

### More examples versus more variety

The size arguments change how many examples are generated. They do not change the kinds of inputs:

- `WORDS` in `dataset.py` supplies the lowercase vocabulary.
- `rng.randint(2, 7)` chooses between 2 and 7 words per sentence.
- `if len(text) > 55` rejects sentences longer than 55 characters.
- Each target is generated with `text.upper()`.

Expand `WORDS` to introduce more vocabulary. To teach other patterns, such as random strings or mixed-case inputs, change `make_split()` to generate those patterns too. Keep inputs within the tokenizer's supported characters and the model's maximum sequence length, allowing room for special tokens.

For the unfamiliar-text problem in section 9, adding varied character patterns may help more than producing many more sentences from the same word list. Compare validation results to judge whether the change helps.

## 11. Quick reference

```text
Dataset size          = 12,000
Epochs                = 30
Batch size            = 64
Batches per epoch     = ceil(12,000 / 64) = 188
Total examples seen   = 12,000 × 30 = 360,000
Total optimizer steps = 188 × 30 = 5,640
```

**In one sentence:** An epoch is a full dataset pass; a batch is a group of examples processed together; and a weight update is the optimizer adjusting the model's parameters after computing gradients.
