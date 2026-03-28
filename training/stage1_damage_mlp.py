"""Stage 1: Prithvi damage MLP training.

Trains a 2-layer MLP (768 → 256 → 4) on synthetic data (scaffold for xBD).
Computes real weighted_f1 on a held-out test split instead of hardcoding metrics.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import weighted_f1


def compute_weighted_f1(model: nn.Module, dataloader: DataLoader, num_classes: int = 4) -> float:
    """Compute real weighted F1 score on a dataset."""
    model.eval()
    # Per-class counts: [TP, FP, FN, support]
    counts = {c: [0, 0, 0, 0] for c in range(num_classes)}

    with torch.no_grad():
        for bx, by in dataloader:
            logits = model(bx)
            preds = logits.argmax(dim=1)
            for c in range(num_classes):
                pred_c = (preds == c)
                true_c = (by == c)
                tp = (pred_c & true_c).sum().item()
                fp = (pred_c & ~true_c).sum().item()
                fn = (~pred_c & true_c).sum().item()
                support = true_c.sum().item()
                counts[c][0] += tp
                counts[c][1] += fp
                counts[c][2] += fn
                counts[c][3] += support

    classes_data = [(counts[c][0], counts[c][1], counts[c][2], counts[c][3]) for c in range(num_classes)]
    return weighted_f1(classes_data)


def main() -> None:
    torch.manual_seed(42)
    n_total = 2000
    n_test = 400
    n_train = n_total - n_test

    x = torch.randn(n_total, 768)
    y = torch.randint(0, 4, (n_total,))

    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    print("Stage 1 Damage MLP: Training...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_dl:
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        # Evaluate on test set each epoch
        test_f1 = compute_weighted_f1(model, test_dl)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, test_weighted_f1={test_f1:.4f}")

    # Final evaluation
    final_f1 = compute_weighted_f1(model, test_dl)
    print(f"Final test weighted_f1 = {final_f1:.4f}")

    out_dir = Path("models/weights/prithvi")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "damage_mlp.pt"
    torch.save(model.state_dict(), ckpt)

    report = {
        "checkpoint": str(ckpt),
        "weighted_f1": round(final_f1, 4),
        "epochs_trained": epochs,
        "train_samples": n_train,
        "test_samples": n_test,
        "note": "Real weighted_f1 computed on held-out test split. "
                "Using synthetic data scaffold; replace with xBD dataloader for real run.",
    }
    rp = Path("results/stage1_damage_mlp.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {ckpt}")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
