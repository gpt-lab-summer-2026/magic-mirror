"""
Fit AnchorNet to the labelled cutouts, and say whether it beat the heuristics.

Every epoch prints the error in reference spans for both halves of the split,
next to what auto_anchors scores on the same validation photos - because a model
that cannot beat the code it replaces is not worth loading.

Usage:
    python train_anchors.py [folder]     # defaults to dataset/
"""
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from anchor_model import CHECKPOINT, AnchorNet, span_error
from garment_dataset import DATASET_DIR, GarmentPoints
from score_anchors import baseline

EPOCHS = 60
BATCH_SIZE = 16
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4    # the backbone already knows cloth; the head starts knowing nothing
WEIGHT_DECAY = 1e-4


def run_epoch(model, loader, loss_fn, device, optimiser=None):
    """One pass over a loader. With an optimiser it trains, without one it scores."""
    model.train(optimiser is not None)
    loss_total = error_total = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(optimiser is not None):
            predicted = model(images)
            loss = loss_fn(predicted, targets)
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        loss_total += loss.item() * len(images)
        error_total += span_error(predicted.detach(), targets).item() * len(images)
    return loss_total / len(loader.dataset), error_total / len(loader.dataset)


def main(directory):
    train_set = GarmentPoints(directory)
    val_set = GarmentPoints(directory, validation=True)
    if not len(train_set) or not len(val_set):
        sys.exit(f"{directory}/ has {len(train_set)} training and {len(val_set)} validation "
                 f"photos - label more with python label_garments.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(train_set)} training, {len(val_set)} validation, on {device}")
    print(f"heuristics score {baseline(val_set):.3f} spans on the validation half\n")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = AnchorNet().to(device)
    optimiser = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": BACKBONE_LR},
         {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)
    loss_fn = nn.SmoothL1Loss()

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        _, train_error = run_epoch(model, train_loader, loss_fn, device, optimiser)
        _, val_error = run_epoch(model, val_loader, loss_fn, device)
        schedule.step()

        saved = ""
        if val_error < best:
            # Best on validation, not last: with a few hundred garments the last
            # epoch is usually a little overfitted to the training half.
            best, saved = val_error, "   <- saved"
            torch.save(model.state_dict(), CHECKPOINT)
        print(f"epoch {epoch:3d}   train {train_error:.3f}   val {val_error:.3f} spans{saved}")

    print(f"\nbest {best:.3f} spans, in {CHECKPOINT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET_DIR)
