import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from t4fnet.model import T4FNet


def main():
    parser = argparse.ArgumentParser(description="Check T4FNet checkpoint compatibility")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--medsam-base-checkpoint", required=True)
    parser.add_argument("--medsam-checkpoint", default=None)
    args = parser.parse_args()

    model = T4FNet(
        medsam_base_checkpoint=args.medsam_base_checkpoint,
        medsam_checkpoint=args.medsam_checkpoint,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Checkpoint is compatible. Parameters: {total:,}; trainable: {trainable:,}")


if __name__ == "__main__":
    main()
