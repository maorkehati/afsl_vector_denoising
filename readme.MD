# 🧠 Binary Vector Denoising

📄 **Report:** [report.pdf](./report.pdf)  
📦 **Predictions:** [test_predictions.pickle](./test_predictions.pickle)

A flexible and modular framework for experimenting with signal-based binary classification models — from classical MAP estimators to neural networks — using k-fold validation and extensible data/model pipelines.

---

## 📁 Project Structure

```
sct/
├── configs/
│   ├── templates/              # Model-specific template configs
│   └── grid_search/            # Grid search hyperparameter spec files
├── data/                       # Input data files (train/test pickle format)
├── dataloaders/               # Utilities to load data in different formats
│   ├── basic_data_loader.py
│   ├── torch_dataset_loader.py
│   └── dataloader_builder.py
├── metrics/                   # Evaluation metrics (e.g., accuracy, BCE, ECE)
├── models/                    # Model definitions (MAP, GMM, Linear, Conv1D)
├── runners/                   # Training, grid search, k-fold logic
├── utils/                     # YAML + dict utilities
├── train.py                   # Entrypoint script
└── explore.ipynb              # Notebook for analysis
```

---

## 🧪 Models Supported

- **MAPEstimator** – Classical Maximum A Posteriori estimator using Gaussian statistics.
- **Bimodal_GMM_MAP_Estimator** – Uses a 2-component GMM per class.
- **LinearRegression** – Learns a shared linear filter over a local window.
- **LocalMultivariateMAPEstimator** – Uses multivariate Gaussian distributions over local Y windows.
- **Conv1DNN** – Configurable 1D CNN with nonlinearities and depth.

---

## 📦 Data Format

Expected format for `basic` mode:
- Pickled dictionary with keys `"X"` and `"Y"`
- `X`: `[N, S]` binary labels
- `Y`: `[N, S]` real-valued inputs (possibly noisy/corrupted)

---

## ⚙️ Configuration

Training is driven by YAML config files. For example:

```yaml
experiment_name: conv_baseline
seed: 42
model:
  type: Conv1DNN
  params:
    L: 5
    depth: 3
    width: 64
    activation: relu
data:
  type: basic
  path: data/train.pickle
eval:
  type: basic
  test_path: data/test.pickle
  metrics:
    - accuracy
    - bce
    - ece
training:
  epochs: 20
  lr: 0.001
  loss: bce
```

Use files in `configs/templates/` as starting points.

---

## 🧮 Supported Metrics

| Metric Name         | Description                                      | Notes                                      |
|---------------------|--------------------------------------------------|--------------------------------------------|
| `accuracy`          | Percent of correct predictions                   | -                                          |
| `f1_score`          | Harmonic mean of precision and recall            | -                                          |
| `precision`         | Correct positives over all predicted positives   | -                                          |
| `recall`            | Correct positives over all actual positives      | -                                          |
| `roc_auc`           | Area under the ROC curve                         | -                                          |
| `hamming_distance`  | Percent of mismatched positions                  | Equal to `1 - accuracy`                    |
| `bce`               | Binary Cross Entropy loss                        | ✅ Only for **logit-based** models         |
| `brier`             | Brier score (mean squared error of probs)        | ✅ Only for **logit-based** models         |
| `ece`               | Expected Calibration Error (prob-confidence gap) | ✅ Only for **logit-based** models         |

---

## 🚀 Running Training

### Option 1: Standard Training

Train using a single static config:

```bash
python train.py configs/templates/conv1dnn.yml
```

### Option 2: Hyperparameter Grid Search

Train using a **grid search config** that performs multiple experiments with varied hyperparameters:

```bash
python train.py configs/grid_search/map_shift.yml
```

Example grid search config:

```yaml
config: configs/templates/map_shift-1.yml
search:
  model.params.shift: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
```

Explanation:
- `config` points to the base YAML template for the experiment.
- `search` defines a dictionary of keys to sweep. The keys are in **dotted notation** (i.e., they refer to nested YAML keys).
- Each value is a list of hyperparameter values to try.

---

## 🔁 K-Fold Cross Validation

All training runs (single or grid search) use **5-fold cross-validation**. Results are averaged across folds and printed at the end.

Sample output:

```
📊 Final Averaged Metrics Across All Folds:
────────────────────────────────────────────
⭐ Accuracy  : 0.8712
⭐ F1_score  : 0.8734
⭐ Bce       : 0.2741
⭐ Ece       : 0.0487
```

---

## 🔍 Hyperparameter Grid Search Summary

After all configurations are tested, a summary table is printed with metrics for each parameter combination.

Example:

| model.params.L | accuracy  | bce     |
|----------------|-----------|---------|
| 3              | 0.8471    | 0.6032  |
| 4              | 0.8539    | 0.5881  |
| 5              | 🟩 0.8615 | 🟩 0.5621 |
| 6              | 0.8602    | 0.5698  |
| 7              | 0.8593    | 0.5773  |

_(Best value in each metric is highlighted)_

---

## 🔧 Customization

- Add new models in `models/`, and register them in `model_builder.py`.
- Add new metrics in `metrics/` and register in `load_metric()`.
- Modify the torch training loop in `runners/torch_training_loop.py`.

---
