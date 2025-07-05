import torch
from dataloaders.dataloader_builder import build_dataloader
from models import build_model
from runners.torch_training_loop import train_torch_model
import pickle

def save_predictions(predictions, filename="predictions.pickle"):
    """
    Saves the predictions tensor to a pickle file.

    Args:
        predictions (Tensor or ndarray): Tensor of shape [B, S]
        filename (str): Output file name
    """
    with open(filename, "wb") as f:
        pickle.dump(predictions, f)
        
    print(f"Predictions saved to {filename}")
        
def run_full_train(config, batch_size=128, seed=42):
    """
    Trains a model on the full training set using the specified config.
    Optionally performs test-time inference if a test set is provided.

    Args:
        config (dict): Configuration dictionary.
        batch_size (int): Batch size for training.
        seed (int): Random seed.

    Returns:
        dict: Training results (metrics, etc.)
        np.ndarray or None: Test set predictions if test set is given.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_type = config["data"]["type"]
    train_data = build_dataloader(mode=data_type, path=config["data"]["path"], batch_size=batch_size)

    if data_type == "basic":
        X_train, Y_train = train_data.dataset.tensors
        model = build_model(config["model"], X=X_train, Y=Y_train)
        with torch.no_grad():
            preds = model(Y_train)
        X_val = X_train  # dummy, for metric compatibility

    elif data_type == "torch":
        model = build_model(config["model"])
        preds, X_val = train_torch_model(model, train_data, val_loader=None, config=config)

    else:
        raise ValueError(f"Unsupported dataloader mode: {data_type}")

    # Compute training metrics
    metric_names = config["eval"]["metrics"]
    from metrics import load_metric
    final_results = {}
    print("\n📈 Training Set Metrics:")
    for name in metric_names:
        fn = load_metric(name)
        score, fmt, _, _ = fn(preds, X_val)
        final_results[name] = score
        print(f"   ✅ {name.capitalize():<10}: {fmt(score)}")

    # Optional: Run on test set
    test_preds = None
    if "test" in config:
        print("\n🔮 Running inference on test set...")
        test_cfg = config["test"]
        test_data = build_dataloader(
            mode=test_cfg["type"],
            path=test_cfg["test_path"],
            batch_size=batch_size,
            shuffle=False,
            is_test=True
        )

        model.eval()
        device = next(model.parameters()).device
        test_preds = []
        for Y in test_data:
            Y = Y.to(device)
            with torch.no_grad():
                out = model(Y, binary=True)
            test_preds.append(out.cpu())

        test_preds = torch.cat(test_preds, dim=0).numpy()
        
        save_predictions(test_preds, filename=test_cfg.get("output_file", "test_predictions.pickle"))

    return final_results, test_preds
