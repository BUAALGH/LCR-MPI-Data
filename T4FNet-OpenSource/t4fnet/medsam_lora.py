from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry


class MPIFeatureBooster(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = self.layer2(self.layer1(x))
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)


class AutoPromptGenerator(nn.Module):
    def __init__(self, threshold_percentile=15, padding=16):
        super().__init__()
        self.threshold_percentile = threshold_percentile
        self.padding = padding

    def forward(self, x):
        batch, _, height, width = x.shape
        boxes = []
        for index in range(batch):
            image = x[index, 0].detach().cpu().numpy()
            scaled = (
                (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
            ).astype(np.uint8)
            _, otsu = cv2.threshold(
                scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            threshold = image.max() * (self.threshold_percentile / 100.0)
            percentile = (image > threshold).astype(np.uint8) * 255
            binary = np.maximum(otsu, percentile)
            count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if count > 1:
                label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                x1 = stats[label, cv2.CC_STAT_LEFT]
                y1 = stats[label, cv2.CC_STAT_TOP]
                x2 = x1 + stats[label, cv2.CC_STAT_WIDTH]
                y2 = y1 + stats[label, cv2.CC_STAT_HEIGHT]
                x1 = max(0, x1 - self.padding)
                y1 = max(0, y1 - self.padding)
                x2 = min(width, x2 + self.padding)
                y2 = min(height, y2 + self.padding)
                scale = 1024 / 64
                box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
            else:
                box = [0, 0, 1024, 1024]
            boxes.append(box)
        return torch.tensor(boxes, dtype=torch.float32, device=x.device)


class MPISegMedLoRA(nn.Module):
    def __init__(self, checkpoint_path, tuned_checkpoint=None, lora_rank=4):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MedSAM base checkpoint not found: {checkpoint_path}")
        self.sam_model = sam_model_registry["vit_b"](checkpoint=None)
        base_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.sam_model.load_state_dict(base_state, strict=True)
        for parameter in self.sam_model.parameters():
            parameter.requires_grad = False
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        self.sam_model.image_encoder = get_peft_model(
            self.sam_model.image_encoder, config
        )
        self.upsampler = MPIFeatureBooster(in_channels=1, out_channels=3)
        self.auto_prompt = True
        self.prompt_generator = AutoPromptGenerator()
        for parameter in self.sam_model.mask_decoder.parameters():
            parameter.requires_grad = True
        if tuned_checkpoint:
            checkpoint = torch.load(tuned_checkpoint, map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint)
            self.load_state_dict(state, strict=False)

    def forward(self, x, box_prompt=None):
        if box_prompt is None:
            box_prompt = self.prompt_generator(x)
        image_embedding = self.sam_model.image_encoder(self.upsampler(x))
        box_prompt = box_prompt.to(image_embedding.device)
        sparse, dense = self.sam_model.prompt_encoder(
            points=None, boxes=box_prompt, masks=None
        )
        target_batch = image_embedding.shape[0]
        if sparse.shape[0] != target_batch:
            repeat = target_batch // sparse.shape[0]
            if repeat * sparse.shape[0] != target_batch:
                raise ValueError("Prompt batch does not match image batch")
            sparse = sparse.repeat_interleave(repeat, dim=0)
            dense = dense.repeat_interleave(repeat, dim=0)
        dense_pe = self.sam_model.prompt_encoder.get_dense_pe()
        masks = []
        iou_predictions = []
        for index in range(target_batch):
            mask, iou = self.sam_model.mask_decoder(
                image_embeddings=image_embedding[index:index + 1],
                image_pe=dense_pe if dense_pe.shape[0] == 1 else dense_pe[index:index + 1],
                sparse_prompt_embeddings=sparse[index:index + 1],
                dense_prompt_embeddings=dense[index:index + 1],
                multimask_output=False,
            )
            mask = torch.sigmoid(mask)
            mask = F.interpolate(
                mask, size=(64, 64), mode="bilinear", align_corners=False
            )
            if not self.training:
                mask = (mask > 0.5).float()
            masks.append(mask)
            iou_predictions.append(iou)
        return torch.cat(masks, dim=0), torch.cat(iou_predictions, dim=0)

