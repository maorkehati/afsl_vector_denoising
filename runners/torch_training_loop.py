import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_loss_function(name: str) -> nn.Module:
    """
    Returns the appropriate PyTorch loss function based on the name.

    Args:
        name (str): Name of the loss function.

    Returns:
        nn.Module: Corresponding loss function module.

    Raises:
        ValueError: If the loss name is unsupported.
    """
    name = name.lower()
    if name in ["bce", "binary_crossentropy", "bcewithlogits"]:
        return nn.BCEWithLogitsLoss()
    elif name in ["mse", "l2", "mean_squared_error"]:
        return nn.MSELoss()
    elif name in ["mae", "l1", "mean_absolute_error"]:
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: '{name}'")


def train_torch_model(model, train_loader, val_loader, config, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains a model using torch DataLoader and returns predictions on the validation set.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        config (dict): Config containing optimizer and training parameters.
        device (str): Device to train on.

    Returns:
        preds (Tensor): Predictions on the full validation set.
        targets (Tensor): Ground truth labels.
    """

    model.to(device)
    model.train()

    training_cfg = config.get("training", {})
    lr = training_cfg.get("learning_rate", 1e-3)
    lr = float(lr)  # Ensure learning rate is a float
    epochs = training_cfg.get("epochs", 10)
    loss_name = training_cfg.get("loss", "bce")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = get_loss_function(loss_name)

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            logits = model(Y_batch)
            loss = criterion(logits, X_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📉 Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    # --- Evaluate on validation set ---
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            logits = model(Y_batch)
            pred = (logits > 0).float()  # binary prediction threshold at 0
            preds.append(pred.cpu())
            targets.append(X_batch.cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return preds, targets
