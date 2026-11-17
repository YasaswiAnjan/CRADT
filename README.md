# CRADT — Compute-Regularized Adaptive Depth Transformer

An adaptive-depth Transformer for sentiment classification. Unlike a fixed-depth model, CRADT lets **each sample halt at a different layer**, and adds a compute penalty to the training objective so shallower paths are actively rewarded. When task loss stagnates, the model grows a new layer (warm-started by copying the last one) instead of being retrained from scratch.

The entire experiment — model, training loops, baselines, ablations, inference demo, and dashboard generation — lives in a single executable Colab notebook, `CRADT.ipynb`. The committed notebook retains its executed outputs, so the numbers below can be checked against the saved logs.

## Contents

| File | Description |
|---|---|
| `CRADT.ipynb` | Full implementation and experiment pipeline (one code cell, executed; outputs preserved) |
| `requirements.txt` | Python dependencies for running outside Colab |

## Method

**Halting module.** After each transformer layer, a two-layer MLP with GELU and sigmoid predicts a per-sample halt probability `p ∈ (0, 1)` from the mean-pooled hidden state. Its output bias is initialised to `-1.0` so halting starts off unlikely.

**Differentiable depth.** The halting decision is discrete, so the loss uses a soft surrogate:

```
soft_depth = Σ_ℓ  p_ℓ · (ℓ + 1) / L        (normalised to [0, 1])
```

Hard integer depths are tracked separately for the histograms and efficiency figures.

**Objective.**

```
L = L_task + λ · C(d̄)        with λ = 0.05
```

**Structured growth.** A `StagnationDetector` adds a layer when task-loss improvement stays below `δ = 0.15` for `patience = 1` epoch, up to `max_layers = 4`. `expand_layer()` deep-copies the final layer and its halt head (resetting the halt bias), then the optimizer/scheduler are rebuilt. This avoids the loss spike a random new layer would cause.

**Baselines.** Fixed-depth Transformers with 1, 2 and 4 layers, plus DistilBERT-base-uncased with only its last transformer block fine-tuned (all other parameters frozen).

**Datasets.** SST-2 (GLUE), IMDB and Yelp Polarity, 3,000 training examples each, tokenized with `bert-base-uncased` at `max_len = 64`.

## Configuration

| Parameter | Value |
|---|---|
| `seed` | 42 |
| `device` | CUDA if available, else CPU |
| `epochs` | 10 |
| `batch_size` | 16 |
| `learning_rate` | 1e-4 (cosine annealing, η_min 1e-5) |
| `d_model` / `nhead` / `dim_feedforward` | 128 / 4 / 256 |
| `init_layers` → `max_layers` | 2 → 4 |
| `max_len` | 64 |
| `train_samples` | 3,000 per dataset |
| `lambda_compute` (λ) | 0.05 |
| `halt_threshold` (τ) | 0.5 |
| `growth_patience` / `growth_min_delta` | 1 / 0.15 |

## Results

Numbers below are copied from the executed notebook's saved stdout (Colab, T4 GPU).

### SST-2, final epoch

| Model | Layers | Final train acc | Params |
|---|---:|---:|---:|
| Fixed-1L | 1 | 79.3% | 4,047,746 |
| Fixed-2L | 2 | 85.6% | 4,180,226 |
| Fixed-4L | 4 | 92.3% | 4,445,186 |
| DistilBERT (last block tuned) | 6 | 99.2% | 7,089,410 |
| **CRADT** | 2 → 4 | **92.0%** | 4,478,470 |

CRADT SST-2 evaluation: **94.00% accuracy**, 0.421 ms/sample, 441.2 MB peak GPU memory. It grew from 2 to 4 layers at epochs 2 and 4. Initial parameter count was 4,196,868.

### Multi-dataset evaluation

| Dataset | Train acc | Eval acc | Avg. depth | Efficiency |
|---|---:|---:|---:|---:|
| SST-2 | 92.0% | 94.0% | 3.45 / 4 | 86% |
| IMDB | 100.0% | 100.0% | 3.45 / 4 | 86% |
| Yelp Polarity | 84.4% | 85.1% | 3.45 / 4 | 86% |

### Ablation study

| Variant | Accuracy | Final layers |
|---|---:|---:|
| Full CRADT | 97.07% | 4 |
| No compute penalty (λ = 0) | 97.07% | 4 |
| No halting (τ = 1.0) | 97.07% | 4 |
| No expansion (fixed 2L) | 96.10% | 2 |
| "Random depth" (λ = 0, τ = 1) | 96.13% | 2 |

The notebook's complexity summary reports 16,777.2K FLOPs/sample at full depth, 33.8 s total CRADT training time and 0.0% FLOPs saved; see the caveats below for why that last figure differs from the 86% above.

The pipeline also writes a 12-panel `cradt_dashboard.png` covering loss/accuracy curves, compute cost, layer growth, depth distribution, ablations, multi-dataset results, a theoretical bias–variance plot, an efficiency table, accuracy-vs-compute, and training time.

## Running it

**Colab:** open `CRADT.ipynb`, set the runtime to GPU (T4), and run all cells. The first run downloads the GLUE/SST-2, IMDB and Yelp datasets and the `bert-base-uncased` / `distilbert-base-uncased` weights.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook CRADT.ipynb
```

A CUDA GPU is recommended; the code falls back to CPU but the 10-epoch, multi-dataset run is much slower there.

## Caveats on the reported numbers

These are properties of the committed code, not interpretations added later:

- **Evaluation uses the training split.** `build_dataloader` loads `train[:3000]` and the same DataLoader is passed to `evaluate()`. The "eval" accuracies are therefore in-sample, which explains the 100% on IMDB and the high ablation numbers. There is no held-out test set.
- **The 100% efficiency figure is a measurement artifact.** The final complexity summary runs after `run_inference()`, which calls `reset_depth_history()` and pushes six sentences through the model; all six took the full 4 layers. The per-dataset training figure of 3.45/4 (86%) comes from the depth history recorded during training and is the representative one.
- **The "Random depth" ablation is not random.** The forward pass has no stochastic depth; that row is λ = 0 with τ = 1 and expansion disabled, and differs from "No halting" only in that expansion is off.
- **τ = 1.0 means "never halt"**, since a sigmoid output is strictly below 1. The `run_ablation` docstring's comment about τ = 0 halting at layer 1 does not match the variant list, which actually uses τ = 1.0.
- **The dashboard's bias–variance panel is a theoretical illustration**, not measured from the runs.
