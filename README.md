# GPT Language Model — Kernel Code Generator

A character-level GPT transformer trained on Linux kernel C source code.  
Built from scratch in PyTorch. No external model weights. No fine-tuning. Pure training from zero.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Setup & Requirements](#setup--requirements)
5. [How to Run](#how-to-run)
6. [Configuration](#configuration)
7. [What to Expect as Output](#what-to-expect-as-output)
8. [Scaling Laws — And Where Your Model Sits](#scaling-laws--and-where-your-model-sits)
9. [Known Limitations](#known-limitations)

---

## What This Project Does

This project trains a small GPT-style transformer model on Linux kernel C source code and then generates new kernel-like C code character by character — infinitely — in a separate terminal window.

It is a learning project. The goal is not to produce production-quality code, but to understand how language models learn patterns from text, and to see that process happen live on your own machine.

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
- At generation time, it predicts the next character, then the next, then the next — forever

It is the same core architecture as GPT, just much smaller and trained on much less data.

**The pipeline in order:**

```
cleaned.txt
    ↓
Characters encoded as integers
    ↓
Model trains on sequences of 128 characters at a time
    ↓
Every 200 steps: loss is measured and printed
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

No other dependencies are needed. The project uses only PyTorch and Python standard library modules.

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

1. Print a startup banner with device info
2. Load and report stats on your dataset
3. Build the model and print parameter count
4. Train for 3000 steps, printing progress every 200 steps
5. Automatically open a new CMD window with infinite generation when done

**To stop generation:** close the CMD window or press `Ctrl+C` inside it.  
**To stop training early:** press `Ctrl+C` in the main window.

---

## Configuration

All hyperparameters are set at the top of `transformer.py`. The recommended settings for a CPU-only machine with limited RAM are:

```python
batch_size    = 16      # How many sequences to train on at once
block_size    = 128     # How many characters the model sees at once
max_iters     = 3000    # Total training steps
eval_interval = 200     # Print progress every N steps
eval_iters    = 50      # How many batches to average for loss estimate
learning_rate = 3e-4    # How fast the model updates
n_embd        = 128     # Size of internal representations
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
dropout       = 0.2     # Regularization — prevents memorization
```

**Parameter count with these settings: ~0.3M parameters**

If you have a CUDA GPU available, you can scale up:

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

The model is trained on 117,076 characters of Linux kernel C code. After training, the generated output will look something like this:

```c
static int module_get(struct module *mod)
{
    if (!mod->state == MODULE_STATE_LIVE)
        return -ENOENT;
    mutex_lock(&module_mutex);
    list_for_each_entry(mod, &modules
```

**It will look like real kernel C.** Keywords, brackets, struct patterns, function signatures — all plausible. But the logic will be wrong and it will not compile. This is expected and normal for a model of this size trained on this little data.

The more the loss drops during training, the more coherent the output becomes.

| Loss Value | What the output looks like         |
|------------|------------------------------------|
| 4.0+       | Random characters                  |
| 2.5 - 3.5  | Recognizable keywords, messy       |
| 1.8 - 2.5  | Plausible structure, wrong logic   |
| Below 1.8  | Coherent style, still not runnable |

---

## Scaling Laws — And Where Your Model Sits
<img width="1029" height="705" alt="Screenshot 2026-03-17 171921" src="https://github.com/user-attachments/assets/8d6ab428-1588-4d2b-9009-469ad2c1ec14" />


### What Are Scaling Laws?

Scaling laws describe a simple but powerful observation about language models:

> **The more parameters, the more data, and the more compute you use — the better the model gets. And this relationship is predictable.**

This was formally studied by researchers at OpenAI (the Chinchilla paper, 2022). The key finding was:

- Model size (parameters) and dataset size should grow together
- If you double the parameters but keep the dataset the same, you get diminishing returns
- The optimal ratio is roughly **20 tokens of training data per parameter**

### The Three Axes of Scaling

```
Parameters (N)  →  How much the model can remember
Data (D)        →  How much it has learned from
Compute (C)     →  Parameters × Data × Training steps
```

All three need to grow together. Improving only one gives limited gains.

### Where Your Model Sits

Here is how your two configurations compare against each other and against real-world models:

```
Model                   Parameters    Data          Quality
─────────────────────────────────────────────────────────────
Your model (CPU)        0.3M          117K chars    Learning shapes
Your model (full)       10.8M         117K chars    Overfitting risk
GPT-2 Small             117M          ~40GB text    Coherent English
GPT-2 Large             774M          ~40GB text    Strong English
GPT-3                   175B          ~600GB text   Near-human text
```

### My Specific Situation

**Recommended for one with CPU config (0.3M parameters):**

```
Parameters    :  0.3M
Training data :  117,076 characters ≈ ~117K tokens
Optimal data  :  20 × 300,000 = 6M tokens needed for full use
Data ratio    :  117K / 6M = ~2%  ← significantly undertrained
```

**What this means in plain terms:**

Your 0.3M parameter model technically needs about 6 million characters of training data to be fully utilized. You only have 117K. So the model is **data-starved** — it has more capacity than your dataset can fill.

This is actually fine for a learning project. The model will still learn the style and patterns of kernel C code. It just won't reach its theoretical ceiling.

**The full 10.8M parameter model is even more data-starved:**

```
Parameters    :  10.8M
Optimal data  :  20 × 10,800,000 = 216M tokens needed
have      :  117K tokens
Data ratio    :  0.05%  ← severely undertrained
```

Training 10.8M parameters on 117K characters is like hiring 100 experts to read a 10-page document. The capacity is there but there is nothing to fill it with. The model will also take 12-20 hours estimated  on your CPU.

**The practical takeaway:**

For your hardware and dataset, the 0.3M parameter config is the right choice. It will train in 1-2 hours, produce similar quality output to the 10.8M model on this dataset, and not risk crashing your machine due to RAM pressure.

If you want meaningfully better output, the highest-impact thing you can do is **add more training data** — not increase model size. Aim for 1-5 million characters of kernel C code for noticeably better results.

---

## Known Limitations

- **CPU only** — training is slow without a CUDA GPU
- **Small dataset** — 117K characters is not enough to reach the model's full potential
- **Character-level** — the model learns characters, not words or concepts, so it cannot reason about what the code does
- **No memory between generations** — each generation run starts from scratch
- **Output will not compile** — this is a style learner, not a code generator

---
#s# Licence

MIT
---

*Built with PyTorch. Architecture based on Andrej Karpathy's nanoGPT.*
