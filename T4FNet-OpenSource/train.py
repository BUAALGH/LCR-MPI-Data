import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from t4fnet.data import GeneralizationDataset, build_fold_datasets
from t4fnet.losses import ReconstructionLoss
from t4fnet.metrics import compute_metrics
from t4fnet.model import T4FNet
from t4fnet.reproducibility import seed_everything, seed_worker


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def make_loader(dataset, batch_size, shuffle, workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def forward_model(model, inputs):
    prediction, _ = model(inputs[:, 0:1], inputs[:, 1:2])
    return prediction


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total = 0.0
    for inputs, target, _ in tqdm(loader, leave=False, desc="train"):
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            prediction = forward_model(model, inputs)
            loss = criterion(prediction, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total = 0.0
    for inputs, target, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            prediction = forward_model(model, inputs)
            loss = criterion(prediction, target)
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    values = {name: [] for name in ["ssim", "psnr", "nrmse", "fsim", "vif"]}
    for batch in tqdm(loader, leave=False, desc="test"):
        inputs, target = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            prediction = forward_model(model, inputs)
        metrics = compute_metrics(prediction, target)
        for name, value in metrics.items():
            values[name].append(value)
    return {name: float(np.mean(items)) for name, items in values.items()}


def build_model(args, device):
    model = T4FNet(
        medsam_base_checkpoint=args.medsam_base_checkpoint,
        medsam_checkpoint=args.medsam_checkpoint,
        chebyshev_blocks=args.chebyshev_blocks,
    )
    return model.to(device)


def train_fold(fold_index, args, run_dir, device):
    train_set, val_set, test_set = build_fold_datasets(
        args.data_root,
        fold_index,
        seed=args.seed,
        use_augmented=args.use_augmented,
    )
    train_loader = make_loader(
        train_set, args.batch_size, True, args.num_workers,
        args.seed + fold_index,
    )
    val_loader = make_loader(
        val_set, 1, False, args.num_workers, args.seed + 100 + fold_index
    )
    test_loader = make_loader(
        test_set, 1, False, args.num_workers, args.seed + 200 + fold_index
    )
    generalization_loader = make_loader(
        GeneralizationDataset(Path(args.data_root) / "Generalization"),
        1, False, 0, args.seed + 300 + fold_index,
    )

    model = build_model(args, device)
    criterion = ReconstructionLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
    )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    fold_dir = run_dir / f"fold{fold_index + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    latest_path = fold_dir / "checkpoint.pth"
    best_path = fold_dir / "best_model.pth"
    history_path = fold_dir / "history.jsonl"
    start_epoch = 1
    best_val = float("inf")
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]

    print(
        f"Fold {fold_index + 1}: train={len(train_set)}, "
        f"val={len(val_set)}, test={len(test_set)}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
            }, latest_path)
        print(
            f"Fold {fold_index + 1} epoch {epoch:03d}/{args.epochs}: "
            f"train={train_loss:.6f}, val={val_loss:.6f}, best={best_val:.6f}"
        )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return {
        "best_val_loss": best_val,
        "ID": evaluate(model, test_loader, device, use_amp),
        "Generalization": evaluate(
            model, generalization_loader, device, use_amp
        ),
    }


def summarize(fold_results):
    summary = {}
    for subset in ["ID", "Generalization"]:
        summary[subset] = {}
        for metric in ["ssim", "psnr", "nrmse", "fsim", "vif"]:
            values = [result[subset][metric] for result in fold_results]
            summary[subset][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    summary["best_val_loss"] = [result["best_val_loss"] for result in fold_results]
    return summary


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="configs/t4fnet_v4_b.json")
    known, _ = pre_parser.parse_known_args()
    defaults = load_config(known.config)
    parser = argparse.ArgumentParser(description="Train T4FNet with five-fold validation")
    parser.add_argument("--config", default=known.config)
    parser.set_defaults(**defaults)
    parser.add_argument("--data-root")
    parser.add_argument("--medsam-base-checkpoint")
    parser.add_argument("--medsam-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chebyshev-blocks", type=int)
    parser.add_argument("--l1-weight", type=float)
    parser.add_argument("--ssim-weight", type=float)
    parser.add_argument("--edge-weight", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--no-augmented", dest="use_augmented", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"t4fnet_v4_b_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    fold_indices = [args.fold - 1] if args.fold else list(range(5))
    results = [train_fold(index, args, run_dir, device) for index in fold_indices]
    summary = summarize(results)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

