# GPT Language Model — Kernel Code Generator

A character-level GPT transformer built from scratch in PyTorch, trained on Linux kernel C source code to generate kernel-style C code character by character. No pre-trained weights. No fine-tuning. Pure architecture and training from zero.
<img width="948" height="773" alt="Screenshot 2026-03-20 114641" src="https://github.com/user-attachments/assets/a24b9b18-817d-41cd-b0af-efa42aeee0b4" />

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Setup & Requirements](#setup--requirements)
5. [How to Run](#how-to-run)
6. [Configuration](#configuration)
7. [What to Expect as Output](#what-to-expect-as-output)
8. [Actual Training Results](#actual-training-results)
9. [Overfitting — What Happened and Why](#overfitting--what-happened-and-why)
10. [How Weights Produce Output](#how-weights-produce-output)
11. [Scaling Laws — And Where Your Model Sits](#scaling-laws--and-where-your-model-sits)
12. [Known Limitations](#known-limitations)

---

## What This Project Does

This project trains a small GPT-style transformer model on Linux kernel C source code and then generates new kernel-like C code character by character — infinitely — in a separate terminal window.

It is a learning project. The goal is not to produce production-quality code, but to understand how language models learn patterns from text and to see that process happen live on your own machine.

---

## Project Structure

```
project/
│
├── transformer.py        # Main training script — GPT model
├── train.py              # Secondary script — simple Bigram model (for comparison)
│
├── config/
│   └── config.py         # Central config: paths, seed, train split
│
├── cleaned.txt           # Training data — Linux kernel C source
├── best_model.pt         # Best weights saved during training (lowest val loss)
│
├── _gen_weights.pt       # Auto-generated after training (deleted on exit)
├── _gen_vocab.pt         # Auto-generated after training (deleted on exit)
└── _gen_worker.py        # Auto-generated after training (deleted on exit)
```

> The three `_gen_*` files are created automatically when training finishes and cleaned up automatically when the main script exits. You do not need to manage them manually.

---

## How It Works

The model is a **character-level transformer**. This means:

- It reads your text file one character at a time
- It learns which characters tend to follow other characters in which contexts
- At generation time it predicts the next character, then the next, then the next — forever

It is the same core architecture as GPT, just much smaller and trained on much less data.

**The pipeline in order:**

```
cleaned.txt
    ↓
Characters encoded as integers (vocab size: 95)
    ↓
Model trains on sequences of 128 characters at a time
    ↓
Every 200 steps: loss is measured and printed
    ↓
Best weights saved to best_model.pt whenever val loss improves
    ↓
After 3000 steps: weights saved, new CMD window opens
    ↓
New window loads weights and streams generated code forever
```

---

## Setup & Requirements

**Python version:** 3.8 or higher

**Install dependencies:**

```bash
pip install torch
```

No other dependencies needed. The project uses only PyTorch and Python standard library modules.

**Your `config/config.py` should look like this:**

```python
cleaned_path = "cleaned.txt"
train_split  = 0.9
seed         = 42
```

---

## How to Run

```bash
python transformer.py
```

That is it. The script will:

1. Print a startup banner with device and timestamp
2. Load and report stats on your dataset
3. Build the model and print parameter count
4. Train for 3000 steps, printing progress every 200 steps
5. Automatically open a new CMD window with infinite generation when done

**To stop generation:** close the CMD window or press `Ctrl+C` inside it.
**To stop training early:** press `Ctrl+C` in the main window.

---

## Configuration

All hyperparameters are at the top of `transformer.py`. The settings below are tuned for a CPU-only machine with limited RAM:

```python
batch_size    = 16      # How many sequences to train on at once
block_size    = 128     # How many characters the model sees at once
max_iters     = 3000    # Total training steps
eval_interval = 200     # Print progress every N steps
eval_iters    = 50      # Batches averaged for loss estimate
learning_rate = 3e-4    # How fast the model updates
n_embd        = 128     # Size of internal representations
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
dropout       = 0.2     # Regularization — prevents memorization
```

**Parameter count with these settings: 0.83M parameters**

If you have a CUDA GPU, you can scale up:

```python
batch_size    = 64
block_size    = 256
n_embd        = 384
n_head        = 6
n_layer       = 6
max_iters     = 5000
eval_iters    = 200
```

**Parameter count with GPU settings: ~10.8M parameters**

---

## What to Expect as Output

The model is trained on 117,076 characters of Linux kernel C source code. After training, generated output looks roughly like:

```c
static int module_get(struct module *mod)
{
    if (!mod->state == MODULE_STATE_LIVE)
        return -ENOENT;
    mutex_lock(&module_mutex);
    list_for_each_entry(mod, &modules
```

It will look like real kernel C — keywords, brackets, struct patterns, function signatures — all plausible. But the logic will be wrong and it will not compile. This is expected and normal for a model of this size trained on this little data.

| Loss Value  | What the output looks like          |
|-------------|-------------------------------------|
| 4.0+        | Random characters                   |
| 2.5 - 3.5   | Recognizable keywords, messy        |
| 1.8 - 2.5   | Plausible structure, wrong logic    |
| Below 1.8   | Coherent style, still not runnable  |

---

## Actual Training Results

Trained on: AMD Ryzen 5 PRO 3500U (CPU only, no GPU)

```
Parameters    : 0.83M
Dataset       : 117,076 characters of Linux kernel C
Vocabulary    : 95 unique characters
Training time : 36 minutes
Best val loss : 2.3924  (reached at step 1400)
Final train   : 0.7820
```

**Full training log:**

```
[    0/3000]   0.0%   train=4.5517   val=4.5657   elapsed=13s     ETA=0s      << best!
[  200/3000]   6.7%   train=2.5428   val=2.8737   elapsed=161s    ETA=2239s   << best!
[  400/3000]  13.3%   train=2.3188   val=2.7972   elapsed=321s    ETA=2083s   << best!
[  600/3000]  20.0%   train=2.0389   val=2.6563   elapsed=479s    ETA=1910s   << best!
[  800/3000]  26.7%   train=1.6357   val=2.6205   elapsed=631s    ETA=1732s   << best!
[ 1000/3000]  33.3%   train=1.4412   val=2.5216   elapsed=782s    ETA=1561s   << best!
[ 1200/3000]  40.0%   train=1.2984   val=2.4317   elapsed=924s    ETA=1384s   << best!
[ 1400/3000]  46.7%   train=1.1895   val=2.3924   elapsed=1069s   ETA=1220s   << best!
[ 1600/3000]  53.3%   train=1.0938   val=2.4409   elapsed=1210s   ETA=1058s
[ 1800/3000]  60.0%   train=1.0317   val=2.4111   elapsed=1352s   ETA=900s
[ 2000/3000]  66.7%   train=0.9991   val=2.4207   elapsed=1487s   ETA=743s
[ 2200/3000]  73.3%   train=0.9201   val=2.4054   elapsed=1615s   ETA=586s
[ 2400/3000]  80.0%   train=0.8548   val=2.4371   elapsed=1750s   ETA=437s
[ 2600/3000]  86.7%   train=0.8339   val=2.4127   elapsed=1888s   ETA=290s
[ 2800/3000]  93.3%   train=0.8137   val=2.4009   elapsed=2021s   ETA=144s
[ 2999/3000] 100.0%   train=0.7820   val=2.4854   elapsed=2159s   ETA=0s

[DONE] Training finished in 2159.5s (36.0 min) | Best val loss: 2.3924
```

---

## Overfitting — What Happened and Why

Looking at the training log, something important happened after step 1400:

```
Step 1400:  train=1.1895   val=2.3924   ← val loss at its lowest (best)
Step 1600:  train=1.0938   val=2.4409   ← val loss starts rising
Step 3000:  train=0.7820   val=2.4854   ← train keeps falling, val keeps rising
```

**This is textbook overfitting.**

The model has two jobs during training — learn general patterns from the data, and not just memorize the training examples. Up to step 1400 it was doing both. After step 1400, the train loss kept falling but the val loss started rising. This means the model stopped learning general patterns and started memorizing the specific training text.

**In plain terms:**

```
Before step 1400 → model is learning kernel C patterns
After  step 1400 → model is memorizing specific lines
                   from cleaned.txt word for word
```

**Why this happened:**

The dataset is too small (117K characters) for even a 0.83M parameter model. The model has more capacity than the data can fill, so after enough steps it starts memorizing instead of generalizing.

**The fix — save best weights, not final weights:**

The current script saves weights at the end of training (step 3000). But the best weights were at step 1400. Add this one change to always keep the best version:

```python
# Inside the eval checkpoint block:
if losses['val'] < best_val_loss:
    best_val_loss = losses['val']
    torch.save(model.state_dict(), 'best_model.pt')  # add this line
    improved = " << best!"
```

Then load `best_model.pt` for generation instead of the final weights. This gives noticeably better output quality.

**Other ways to reduce overfitting:**

- Add more training data — most effective, aim for 1M+ characters
- Increase dropout from 0.2 to 0.3 or 0.4
- Reduce model size further
- Use early stopping — stop training when val loss stops improving

---

## How Weights Produce Output

After training, the model is frozen. The weights file is just a collection of numbers — 0.83 million of them — that encode everything the model learned from your kernel C code.

**The generation loop step by step:**

```
Step 1 — Start with a seed token (zero = start of text)
              ↓
Step 2 — Feed it through all 4 transformer layers
         Each layer does matrix multiplications
         using the saved weight numbers
              ↓
Step 3 — Output is 95 numbers (one per vocab character)
         Each number = probability of that character being next
         e.g.  's' = 0.34   '{' = 0.21   'i' = 0.18
              ↓
Step 4 — Sample randomly from those probabilities
         Higher probability = more likely to be picked
              ↓
Step 5 — That character becomes the new input
         Go back to Step 2
              ↓
Step 6 — Repeat forever
```

**Why output is different every run:**

The sampling step (`torch.multinomial`) picks randomly from the probability distribution. Same weights, different random draws = different output each time. To get reproducible output add `torch.manual_seed(42)` before generation.

**The weights are a compressed snapshot** of every pattern seen in 117K characters of kernel C — stored as 0.83 million floating point numbers.

---

## Scaling Laws — And Where Your Model Sits
<img width="1029" height="705" alt="Screenshot 2026-03-17 171921" src="https://github.com/user-attachments/assets/d376124c-08de-4668-be8d-823e1c05176b" />


### What Are Scaling Laws?

Scaling laws describe a predictable relationship between model size, dataset size, compute, and output quality:

> The more parameters, the more data, and the more compute you use — the better the model gets. And this improvement follows a consistent, measurable curve.

The key finding from research (Chinchilla, 2022) is that model size and dataset size must grow together. The optimal ratio is roughly **20 tokens of training data per parameter.**

### The Three Axes of Scaling

```
Parameters (N)  →  How much the model can remember
Data (D)        →  How much it has learned from
Compute (C)     →  Parameters × Data × Training steps
```

All three need to grow together. Improving only one gives diminishing returns.

### Where Your Model Sits

```
Model                    Parameters    Data             Quality
───────────────────────────────────────────────────────────────
Your model (this run)    0.83M         117K chars        Learning shapes
Your model (full GPU)    10.8M         117K chars        Severely overtrained
GPT-2 Small              117M          ~40GB text        Coherent English
GPT-2 Large              774M          ~40GB text        Strong English
GPT-3                    175B          ~600GB text       Near-human text
```

### Specific Numbers

```
Parameters       :  0.83M
Training data    :  117,076 characters ≈ 117K tokens
Optimal data     :  20 × 830,000 = 16.6M tokens needed
Data you have    :  117K / 16.6M = 0.7% of what is optimal
```

 Model has far more capacity than my dataset can fill. This is exactly why overfitting appeared at step 1400 — the model ran out of new patterns to learn and started memorizing instead.

This does not mean the project failed. The model learned real kernel C style patterns in 36 minutes on a CPU with no GPU. The val loss dropped from 4.56 to 2.39 — that is real learning happening.


---

## Known Limitations

- **CPU only** — no CUDA GPU means training is slow and larger configs are impractical
- **Small dataset** — 117K characters causes overfitting even at 0.83M parameters
- **Overfitting after step 1400** — final weights are not the best weights; see overfitting section
- **Character-level** — the model learns characters not words or concepts, so it cannot reason about what the code does
- **Output will not compile** — this is a style learner, not a functional code generator
- **No memory between runs** — each generation starts from scratch with no context

---

*Built with PyTorch. Architecture inspired by Andrej Karpathy's Let's build GPT: from scratch*
*Trained on Linux kernel source — `kernel/module/core.c` and related files.*
