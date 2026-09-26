import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from t4fnet.data import GeneralizationDataset, InVivoDataset
from t4fnet.model import T4FNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run T4FNet inference")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subset", choices=["generalization", "invivo"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--medsam-base-checkpoint", required=True)
    parser.add_argument("--medsam-checkpoint", default=None)
    parser.add_argument("--output-dir", default="predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T4FNet(
        medsam_base_checkpoint=args.medsam_base_checkpoint,
        medsam_checkpoint=args.medsam_checkpoint,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    root = Path(args.data_root)
    if args.subset == "generalization":
        dataset = GeneralizationDataset(root / "Generalization")
    else:
        dataset = InVivoDataset(root / "InVivo")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch[0].to(device)
            names = batch[2] if args.subset == "generalization" else batch[1]
            prediction, _ = model(inputs[:, 0:1], inputs[:, 1:2])
            np.save(
                output_dir / f"{names[0]}.npy",
                prediction[0, 0].float().cpu().numpy(),
            )


if __name__ == "__main__":
    main()

