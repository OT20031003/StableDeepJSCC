"""Shared PyTorch training, checkpoint, and evaluation utilities."""

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def reconstruct(model: nn.Module, batch: Dict[str, Tensor]) -> Tensor:
    return model(
        batch["image"],
        batch["snr_db"],
        h_real=batch.get("h_real"),
        h_imag=batch.get("h_imag"),
        b_prob=batch.get("b_prob"),
        b_stddev=batch.get("b_stddev"),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.train()
    squared_error = 0.0
    elements = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        output = reconstruct(model, batch)
        loss = F.mse_loss(output, batch["image"])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            squared_error += F.mse_loss(
                output, batch["image"], reduction="sum"
            ).item()
            elements += batch["image"].numel()
    if elements == 0:
        raise ValueError("the training data loader produced no batches")
    return squared_error / elements


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    squared_error = 0.0
    elements = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        output = reconstruct(model, batch)
        squared_error += F.mse_loss(
            output, batch["image"], reduction="sum"
        ).item()
        elements += batch["image"].numel()
    if elements == 0:
        raise ValueError("the evaluation data loader produced no batches")
    return squared_error / elements


def mse_to_psnr(mse: float, data_range: float = 1.0) -> float:
    if mse < 0:
        raise ValueError("MSE cannot be negative")
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(data_range ** 2 / mse)


def save_json(path: str, data: Dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(str(temporary), str(destination))


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Adam] = None,
    epoch: int = 0,
    best_loss: float = float("inf"),
    metadata: Optional[Dict] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(destination))


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[Adam] = None,
) -> Dict:
    source = Path(path)
    if source.suffix.lower() in (".h5", ".hdf5"):
        raise ValueError(
            "Keras .h5 weights cannot be loaded directly; use a PyTorch .pt checkpoint"
        )
    if not source.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(source))
    payload = torch.load(str(source), map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
        payload = {"model_state_dict": state_dict}
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: str,
    history_path: str,
    validation_loader: Optional[DataLoader] = None,
    load_path: Optional[str] = None,
    metadata: Optional[Dict] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> Dict:
    optimizer = Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    best_loss = float("inf")
    if load_path:
        payload = load_checkpoint(load_path, model, device, optimizer)
        start_epoch = int(payload.get("epoch", 0))
        best_loss = float(payload.get("best_loss", best_loss))

    history = {"epoch": [], "loss": []}
    if validation_loader is not None:
        history["val_loss"] = []

    for epoch in range(start_epoch, start_epoch + epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, max_batches=max_train_batches
        )
        monitored_loss = loss
        message = "Epoch: {}, loss={:.8f}".format(epoch + 1, loss)
        if validation_loader is not None:
            val_loss = evaluate(
                model,
                validation_loader,
                device,
                max_batches=max_eval_batches,
            )
            monitored_loss = val_loss
            history["val_loss"].append(val_loss)
            message += ", val_loss={:.8f}".format(val_loss)
        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)

        if monitored_loss < best_loss:
            best_loss = monitored_loss
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch + 1,
                best_loss=best_loss,
                metadata=metadata,
            )
            message += " (saved)"
        print(message)
        save_json(history_path, history)
    return history


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_image(path: str, image: Tensor) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().clamp(0.0, 1.0)
    pil_image = TF.to_pil_image(image)
    pil_image.save(str(destination))
