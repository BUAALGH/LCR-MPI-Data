import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure


class ReconstructionLoss(nn.Module):
    def __init__(self, l1_weight=0.3, ssim_weight=0.7, edge_weight=0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x[None, None])
        self.register_buffer("sobel_y", sobel_y[None, None])

    def forward(self, prediction, target):
        l1 = F.l1_loss(prediction, target)
        ssim = 1.0 - structural_similarity_index_measure(
            prediction, target, data_range=1.0
        )
        if self.edge_weight == 0.0:
            edge = prediction.new_zeros(())
        else:
            pred_x = F.conv2d(prediction, self.sobel_x, padding=1)
            pred_y = F.conv2d(prediction, self.sobel_y, padding=1)
            target_x = F.conv2d(target, self.sobel_x, padding=1)
            target_y = F.conv2d(target, self.sobel_y, padding=1)
            edge = F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
        return (
            self.l1_weight * l1
            + self.ssim_weight * ssim
            + self.edge_weight * edge
        )

