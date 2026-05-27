from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SmartCampusIndoorDataset, make_splits
from src.model import build_model, count_parameters
from src.utils import (
    compute_metrics,
    ensure_dir,
    plot_confusion_matrix,
    plot_curves,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT on MIT Indoor Scenes subset.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--project", type=str, default="csc4005-lab6-mit-indoor-vit")
    parser.add_argument("--run_name", type=str, default="vit_b16_head_only")
    parser.add_argument("--model_name", type=str, default="vit_b_16")
    parser.add_argument("--train_mode", type=str, choices=["head_only", "finetune"], default="head_only")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    config_args, _ = parser.parse_known_args()
    if config_args.config:
        config = json.loads(Path(config_args.config).read_text(encoding="utf-8"))
        parser.set_defaults(**config)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Huấn luyện được cấu hình dùng GPU, nhưng CUDA không khả dụng trên máy hiện tại.")
        return torch.device("cuda")

    return torch.device("cpu")
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics, y_true, y_pred


def main() -> None:
    args = parse_args()
    if args.data_dir is None:
        raise ValueError("Please provide --data_dir pointing to MIT Indoor Scenes subset.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    output_dir = ensure_dir(Path("outputs") / args.run_name)
    save_json(vars(args), output_dir / "config.json")
    print(f"Using device: {device}")

    train_dataset_full = SmartCampusIndoorDataset(
        data_dir=args.data_dir,
        classes=args.classes,
        img_size=args.img_size,
        augment=args.augment,
        max_per_class=args.max_per_class,
    )
    eval_dataset_full = SmartCampusIndoorDataset(
        data_dir=args.data_dir,
        classes=args.classes,
        img_size=args.img_size,
        augment=False,
        max_per_class=args.max_per_class,
    )

    train_idx, val_idx, test_idx = make_splits(
        train_dataset_full,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Reuse identical indices for non-augmented validation/test dataset.
    train_loader = DataLoader(
        train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(eval_dataset_full, val_idx.indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(eval_dataset_full, test_idx.indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    save_json(train_dataset_full.class_to_idx, output_dir / "class_to_idx.json")

    model = build_model(
        model_name=args.model_name,
        train_mode=args.train_mode,
        num_classes=len(train_dataset_full.classes),
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    total_params, trainable_params, trainable_ratio = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                **vars(args),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_ratio": trainable_ratio,
            },
        )

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        train_true, train_pred = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_true.extend(labels.cpu().tolist())
            train_pred.extend(preds.cpu().tolist())

        train_metrics = compute_metrics(train_true, train_pred)
        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)

        if wandb_run is not None:
            wandb_run.log(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_acc={val_metrics['acc']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_dataset_full.class_to_idx,
                    "args": vars(args),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "trainable_ratio": trainable_ratio,
                },
                best_model_path,
            )
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "num_classes": len(train_dataset_full.classes),
        "num_samples": len(train_dataset_full),
        "num_train": len(train_loader.dataset),
        "num_val": len(val_loader.dataset),
        "num_test": len(test_loader.dataset),
    }

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")
    plot_curves(history, output_dir / "curves.png")
    plot_confusion_matrix(y_true, y_pred, train_dataset_full.classes, output_dir / "confusion_matrix.png")

    if wandb_run is not None:
        import wandb

        wandb_run.log(metrics)
        wandb_run.log(
            {
                "curves": wandb.Image(str(output_dir / "curves.png")),
                "confusion_matrix": wandb.Image(str(output_dir / "confusion_matrix.png")),
            }
        )
        wandb_run.finish()

    print("Training completed.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
