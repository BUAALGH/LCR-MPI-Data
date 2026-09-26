import torch
from piq import fsim, vif_p
from torchmetrics.functional import (
    normalized_root_mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def compute_metrics(prediction, target):
    prediction = prediction.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    prediction_rgb = prediction.repeat(1, 3, 1, 1)
    target_rgb = target.repeat(1, 3, 1, 1)
    return {
        "ssim": structural_similarity_index_measure(
            prediction, target, data_range=1.0
        ).item(),
        "psnr": peak_signal_noise_ratio(
            prediction, target, data_range=1.0
        ).item(),
        "nrmse": normalized_root_mean_squared_error(
            prediction, target
        ).item(),
        "fsim": fsim(prediction_rgb, target_rgb, data_range=1.0).item(),
        "vif": vif_p(
            prediction, target, data_range=1.0, sigma_n_sq=2.0
        ).item(),
    }

