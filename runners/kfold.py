import torch
from sklearn.model_selection import KFold
from dataloaders.dataloader_builder import build_dataloader
from models import build_model
from metrics import load_metric
from runners.torch_training_loop import train_torch_model

def run_kfold(
    config,
    k=5,
    batch_size=128,
    seed=42
):
    """
    K-fold cross-validation runner supporting both naive and torch dataloaders.

    Args:
        config (dict): Config with 'data', 'model', and 'eval' keys.
        k (int): Number of folds.
        batch_size (int): Batch size for torch data mode.
        seed (int): Random seed.

    Returns:
        dict: Final averaged metric results {metric_name: scalar score}
    """
    data_type = config["data"]["type"]
    ds = build_dataloader(mode=data_type, path=config["data"]["path"])

    if data_type == "basic":
        X, Y = ds.dataset.tensors
        dataset_len = X.size(0)
    else:
        dataset_len = len(ds.dataset)

    metric_names = config["eval"]["metrics"]
    metric_fns = [load_metric(name) for name in metric_names]

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    score_tracking = {name: [] for name in metric_names}

    print("\n🔁🔁🔁 Starting K-Fold Cross Validation 🔁🔁🔁\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(dataset_len))):
        print("────────────────────────────────────────────────────────────")
        print(f"📂 Fold {fold + 1} / {k}")
        print(f"   📊 Train size: {len(train_idx)} | Val size: {len(val_idx)}")

        train_data = build_dataloader(mode=data_type, path=config["data"]["path"], indices=train_idx, batch_size=batch_size)
        val_data = build_dataloader(mode=data_type, path=config["data"]["path"], indices=val_idx, batch_size=batch_size)

        if data_type == "basic":
            X_train, Y_train = train_data.dataset.tensors
            X_val, Y_val = val_data.dataset.tensors
            model = build_model(config["model"], X=X_train, Y=Y_train)
            with torch.no_grad():
                preds = model(Y_val)

        elif data_type == "torch":
            model = build_model(config["model"])  # Assumes model does not need X/Y at init
            preds, X_val = train_torch_model(model, train_data, val_data, config)

        else:
            raise ValueError(f"Unsupported dataloader mode: {data_type}")

        print("\n📈 Metrics:")
        for name, fn in zip(metric_names, metric_fns):
            score, fmt, agg, direction = fn(preds, X_val)
            score_tracking[name].append(score)
            print(f"   ✅ {name.capitalize():<10}: {fmt(score)}")

        print("✅ Fold complete.")
        print("────────────────────────────────────────────────────────────\n")

    print("📊 Final Averaged Metrics Across All Folds:")
    print("────────────────────────────────────────────")
    final_results = {}
    for name, fn in zip(metric_names, metric_fns):
        scores = score_tracking[name]
        _, fmt, agg, direction = fn(preds, X_val)  # retrieve format and aggregation
        final_score = agg(scores)
        final_results[name] = final_score
        print(f"⭐ {name.capitalize():<10}: {fmt(final_score)}")
    print("────────────────────────────────────────────")
    print("🎉 All folds completed.\n")

    return final_results
