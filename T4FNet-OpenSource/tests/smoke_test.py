import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from t4fnet.model import T4FNet


class DummyMaskGenerator(nn.Module):
    def forward(self, x):
        mask = torch.sigmoid(x)
        return mask, torch.ones(x.shape[0], 1, device=x.device)


def main():
    torch.manual_seed(42)
    model = T4FNet(
        medsam_base_checkpoint=None,
        mask_generator=DummyMaskGenerator(),
    ).eval()
    l3 = torch.rand(1, 1, 64, 64)
    l5 = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        prediction, _ = model(l3, l5)
    assert prediction.shape == (1, 1, 64, 64)
    assert torch.isfinite(prediction).all()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
