# DNN Programming Assignment 2: RNN, LSTM, GRU and Transformer

**Student Name:** Gonuguntla Anand  
**Student ID:** 2026ajo5503  
**Course:** Deep Neural Networks (M.Tech, BITS Pilani WILP)  
**Submission Date:** November 2026  
**Submission Package:** `2026ajo5503_DNN_Assignment.zip`  

---

## 1. Student Variant & Individualisation

- **Student ID:** `2026ajo5503` (Last Digit: `3` $\to$ **Odd Variant**)
- **Task 1.5(c) Primary Individual Ablation:** **Odd Variant** $\to$ LSTM Forget-Gate Bias Initialised to $0.0$ versus $1.0$ at $T = 200$.
  - *Finding:* Initializing $b_f = 0.0$ yields $\sigma(0) = 0.5$, causing cell state memory to decay exponentially ($0.5^{200} \approx 10^{-60}$) and stalling training MSE near the $0.167$ baseline. Initializing $b_f = 1.0$ yields $\sigma(1) \approx 0.73$, maintaining the gradient highway and solving the long-term memory task.
  - *Even Variant Coverage:* The Even variant ablation (Vanilla RNN with gradient clipping norm 1.0 vs None at $T = 200$) is also fully evaluated and discussed in both notebooks and report for comprehensive comparative analysis.
- **Architectural Dimensions:**
  - Recurrent Models (HAR & Adding): Input $d=9$ (HAR) / $d=2$ (Adding), Hidden $h=64$.
  - Transformer Block: Model dimension $d=128$, Feed-Forward expansion $d_{\text{ff}} = 512$ ($4d$), Heads $n_{\text{heads}}=4$, Layers $n_{\text{layers}}=4$.
  - Context Length: $T=128$ tokens for character language modeling on Tiny Shakespeare.
- **Evaluation Seeds:** `42`, `123`, `999` (used systematically for all mean ± std reporting).

---

## 2. Environment & Dependencies

- **Python Version:** 3.10+ (tested on Python 3.10 and 3.13)
- **Framework:** PyTorch 2.x (`torch>=2.0.0`)
- **Core Libraries:** `numpy>=1.24.0`, `pandas>=2.0.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `scikit-learn>=1.2.0`, `reportlab>=4.0.0`
- **Installation:**
  ```bash
  pip install -r requirements.txt
  ```

---

## 3. How to Run Each Notebook

Both notebooks are completely self-contained, contain assert-based verification unit tests directly following each module, use relative file paths, and execute top-to-bottom with **Kernel $\to$ Restart and Run All**.

### Problem 1: Recurrent Architectures (`problem1/problem1.ipynb`)
```bash
cd problem1
# In Google Colab or JupyterLab:
# Runtime -> Change runtime type -> T4 GPU
# Runtime -> Restart and Run All
```
- Downloads and loads UCI HAR dataset automatically.
- Generates the synthetic Adding Problem data in code and validates marker constraints.
- Verifies `MyRNNCell`, `MyLSTMCell`, and `MyGRUCell` from scratch against framework cells within $10^{-5}$.
- Derives parameter counts analytically ($d=9, h=64$) and reconciles with PyTorch's dual-bias implementation.
- Trains and evaluates RNN, LSTM, and GRU on UCI HAR sensor sequences and the Adding Problem ($T \in \{50, 100, 200, 400\}$).
- Computes backpropagation gradient norms $\|\partial \mathcal{L} / \partial h_t\|$ at $T=100$, hidden-size sweeps, and Task 5(c) ablations.
- Saves all 7 plots into `figures/` and appends metrics to `../results.csv`.
- **Expected Runtime on Colab T4 GPU:** ~8–10 minutes.

### Problem 2: Transformer from Scratch (`problem2/problem2.ipynb`)
```bash
cd problem2
# In Google Colab or JupyterLab:
# Runtime -> Change runtime type -> T4 GPU
# Runtime -> Restart and Run All
```
- Implements `scaled_dot_product_attention`, `MultiHeadAttention`, Pre-LayerNorm `TransformerEncoderBlock`, and `SinusoidalPositionalEncoding` from raw tensor operations.
- Validates attention mechanics (row-sums to 1, causal masking, framework match within $10^{-5}$, and exact 3-token hand calculation).
- Evaluates the Encoder Classifier on UCI HAR and ablates positional encoding (Sinusoidal vs Learned vs None).
- Trains the 4-layer, 4-head Causal Language Model on Tiny Shakespeare, logging validation loss (nats) and bits-per-character (BPC).
- Generates 500-character samples at temperatures $0.2$, $0.8$, and $1.5$ into `generated_samples.txt`.
- Generates attention heatmaps, heads ablation ($1, 4, 8$), causal mask removal, and $\mathcal{O}(T^2 d)$ context scaling plots into `figures/`.
- **Expected Runtime on Colab T4 GPU:** ~12–15 minutes.

---

## 4. Submission Folder Structure

```
2026ajo5503_DNN_Assignment/
├── README.md                      ← This documentation file
├── requirements.txt               ← Pinned dependencies
├── report.pdf                     ← Exactly 6-page comprehensive academic report
├── results.csv                    ← 49 benchmark rows across all seeds & variants
├── problem1/
│   ├── problem1.ipynb             ← Self-contained runnable notebook
│   └── figures/                   ← All 7 Problem 1 figures (300 DPI)
│       ├── p1_class_balance.png
│       ├── p1_har_confusion_matrix.png
│       ├── p1_adding_curves_t200.png
│       ├── p1_adding_mse_vs_t.png
│       ├── p1_gradient_norms_t100.png
│       ├── p1_hidden_size_sweep.png
│       └── p1_ablation_clip_forget.png
└── problem2/
    ├── problem2.ipynb             ← Self-contained runnable notebook
    ├── generated_samples.txt      ← 500-char samples at T = 0.2, 0.8, 1.5
    └── figures/                   ← All 5 Problem 2 figures (300 DPI)
        ├── p2_attention_heads_heatmap.png
        ├── p2_lm_learning_curve.png
        ├── p2_heads_ablation.png
        ├── p2_causal_vs_nomask.png
        └── p2_scaling_time_memory.png
```

---

## 5. Assumptions & Implementation Integrity

1. **Subject-Wise Splitting:** 4 held-out validation subjects are chosen dynamically using `np.random.default_rng(seed)` from the 21 training subjects, ensuring zero overlap with the 9 test subjects.
2. **Strict Standardization:** Per-channel mean and standard deviation are computed **strictly from the training subjects** of each seed fold, preventing data leakage into validation or test sets.
3. **From-Scratch Integrity:** `nn.RNNCell`, `nn.LSTMCell`, `nn.GRUCell`, and `F.scaled_dot_product_attention` are strictly used in assert-based verification tests and are never invoked inside custom forward passes.
4. **All Unit Tests Pass:** Every assert check verifies numerical equivalence to framework layers within machine precision ($< 10^{-7}$).
