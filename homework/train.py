import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create log directory with timestamp
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # Create loss function and optimizer
    loss_func = ClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epoch):
        # Clear metrics at the beginning of each epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Training step
            optimizer.zero_grad()  # Clear gradients
            output = model(img)  # Forward pass
            loss = loss_func(output, label)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            # Calculate training accuracy
            train_acc = (output.argmax(dim=1) == label).float().mean().item()
            metrics["train_acc"].append(train_acc)

            global_step += 1

        # Evaluation phase
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                output = model(img)  # Forward pass
                val_acc = (output.argmax(dim=1) == label).float().mean().item()
                metrics["val_acc"].append(val_acc)

        # Log average train and val accuracy to TensorBoard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar("Train/Accuracy", epoch_train_acc.item(), global_step)
        logger.add_scalar("Val/Accuracy", epoch_val_acc.item(), global_step)

        # Print logs on first, last, and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Save the model for grading
    save_model(model)

    # Save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # Optional: additional model hyperparameters
    # parser.add_argument("--num_layers", type=int, default=3)

    # Pass all arguments to train
    train(**vars(parser.parse_args()))
