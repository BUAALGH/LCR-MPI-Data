import torch
import torch.nn as nn
import torch.nn.functional as F

from .medsam_lora import MPISegMedLoRA
from .transformer import PatchEmbed, SATransEncoderV4, TransformerConfig


class ChebyshevMappingModuleV2(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.poly_a = nn.Parameter(0.5 * torch.ones(1))
        self.poly_b = nn.Parameter(torch.ones(1))
        self.poly_c = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=True),
        )

    def forward(self, x):
        physics = self.poly_a * x.square() + self.poly_b * x + self.poly_c
        return self.out_conv(physics) + x


class ChebyshevMappingModuleV2Stack(nn.Module):
    def __init__(self, channels=1, num_blocks=1):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            ChebyshevMappingModuleV2(channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SEFusion(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels * 2 // 4, 0)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2, bias=False),
        )

    def forward(self, l5, l5_pred):
        batch = l5.shape[0]
        joined = torch.cat([l5, l5_pred], dim=1)
        weights = torch.sigmoid(self.fc(self.gap(joined).view(batch, -1)))
        l5_weight = weights[:, 0:1].view(batch, 1, 1, 1)
        pred_weight = weights[:, 1:2].view(batch, 1, 1, 1)
        return l5_weight * l5 + pred_weight * l5_pred


class ConvBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNestDecoderV4(nn.Module):
    def __init__(self, hidden_size, decoder_dim=64):
        super().__init__()
        self.s2_conv = ConvBnReLU(hidden_size, decoder_dim)
        self.s2_deconv = DeconvBlock(decoder_dim, decoder_dim)
        self.s1_conv = ConvBnReLU(hidden_size, decoder_dim)
        self.s1_deconv = DeconvBlock(decoder_dim, decoder_dim)
        self.s1_fuse = ConvBnReLU(decoder_dim * 2, decoder_dim)
        self.s1_final_deconv = DeconvBlock(decoder_dim, decoder_dim)

    def forward(self, skips):
        skip1, skip2 = skips
        y2 = self.s2_deconv(self.s2_conv(skip2))
        skip1_feature = self.s1_deconv(self.s1_conv(skip1))
        y1 = self.s1_fuse(torch.cat([skip1_feature, y2], dim=1)) + skip1_feature
        return self.s1_final_deconv(y1)


class SATransGHV4(nn.Module):
    def __init__(self, config, num_classes=1):
        super().__init__()
        hidden = config.hidden_size
        decoder = config.decoder_dim
        self.patch_embed = PatchEmbed(1, hidden, patch_size=4, img_size=64)
        self.global_skip = nn.Sequential(
            ConvBnReLU(1, decoder),
            ConvBnReLU(decoder, decoder),
        )
        self.encoder = SATransEncoderV4(config, vis=True)
        self.decoder = UNestDecoderV4(hidden, decoder)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.final_fuse = nn.Sequential(
            ConvBnReLU(decoder * 2, decoder),
            ConvBnReLU(decoder, decoder),
        )
        self.head = nn.Sequential(
            ConvBnReLU(decoder, decoder),
            nn.Conv2d(decoder, num_classes, 3, padding=1),
            nn.Sigmoid(),
        )

    def _encode(self, x, soft_mask, return_attn=False):
        tokens, _, _ = self.patch_embed(x)
        return self.encoder(tokens, soft_mask, return_attn=return_attn)

    def _decode(self, x, skips):
        features = self.global_skip(x)
        skips = [
            item.permute(0, 2, 1).view(x.size(0), -1, 16, 16)
            for item in skips
        ]
        y1 = self.decoder(skips)
        output = self.final_fuse(torch.cat([y1, features], dim=1)) + self.gamma * y1
        return self.head(output)

    def forward(self, x, soft_mask):
        return self._decode(x, self._encode(x, soft_mask))

    def forward_with_attn(self, x, soft_mask):
        skips, attention = self._encode(x, soft_mask, return_attn=True)
        return self._decode(x, skips), attention


class T4FNet(nn.Module):
    def __init__(self, medsam_base_checkpoint, medsam_checkpoint=None,
                 chebyshev_blocks=1, mask_generator=None):
        super().__init__()
        if chebyshev_blocks == 1:
            self.chebyshev = ChebyshevMappingModuleV2(channels=1)
        else:
            self.chebyshev = ChebyshevMappingModuleV2Stack(
                channels=1, num_blocks=chebyshev_blocks
            )
        self.fusion = SEFusion(channels=1)
        self.mask_generator = mask_generator or MPISegMedLoRA(
            checkpoint_path=medsam_base_checkpoint,
            tuned_checkpoint=medsam_checkpoint,
            lora_rank=4,
        )
        self.mask_generator.eval()
        for parameter in self.mask_generator.parameters():
            parameter.requires_grad = False
        self.satrans = SATransGHV4(TransformerConfig(), num_classes=1)

    def forward_with_aux(self, l3, l5):
        if l3.shape != l5.shape or l3.ndim != 4 or l3.shape[1:] != (1, 64, 64):
            raise ValueError("Expected matching l3 and l5 tensors with shape [B, 1, 64, 64]")
        l5_pred = self.chebyshev(l3)
        l5_fused = self.fusion(l5, l5_pred)
        with torch.no_grad():
            pixel_mask, _ = self.mask_generator(l3)
        soft_mask = F.adaptive_avg_pool2d(pixel_mask, (16, 16)).flatten(1)
        prediction = self.satrans(l5_fused, soft_mask)
        return prediction, l5_pred, l5_fused, pixel_mask, soft_mask, soft_mask

    def forward(self, l3, l5):
        prediction, _, _, _, _, _ = self.forward_with_aux(l3, l5)
        return prediction, None

    def forward_with_aux_and_attn(self, l3, l5):
        l5_pred = self.chebyshev(l3)
        l5_fused = self.fusion(l5, l5_pred)
        with torch.no_grad():
            pixel_mask, _ = self.mask_generator(l3)
        soft_mask = F.adaptive_avg_pool2d(pixel_mask, (16, 16)).flatten(1)
        prediction, attention = self.satrans.forward_with_attn(l5_fused, soft_mask)
        return (
            prediction, l5_pred, l5_fused, pixel_mask,
            soft_mask, soft_mask, attention,
        )


T2FNetB_v4 = T4FNet

