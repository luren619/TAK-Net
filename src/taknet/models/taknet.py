from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.nn.modules.utils import _pair

from .vit_seg_modeling import Block, DecoderCup, SegmentationHead, np2th
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


class DualModalAdaptiveFusion(nn.Module):
    """DAF: adaptive fusion of aligned B-mode and CEUS features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.bmode_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.ceus_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.dirac_(self.bmode_projection.weight)
        nn.init.zeros_(self.bmode_projection.bias)
        nn.init.dirac_(self.ceus_projection.weight)
        nn.init.zeros_(self.ceus_projection.bias)

    def forward(self, bmode_feat: torch.Tensor, ceus_feat: torch.Tensor) -> torch.Tensor:
        bmode = self.bmode_projection(bmode_feat)
        ceus = self.ceus_projection(ceus_feat)
        gate = self.sigmoid(self.local_att(bmode + ceus) + self.global_att(bmode + ceus))
        return 2.0 * bmode * gate + 2.0 * ceus * (1.0 - gate)


class NormalizedResidualMerge(nn.Module):
    """Normalized residual merge used by MCE."""

    def __init__(
        self,
        num_levels: int,
        init_context_weight: float = 0.05,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        init = torch.tensor(
            [[1.0, float(init_context_weight)] for _ in range(num_levels)], dtype=torch.float32
        )
        self.weights = nn.Parameter(init)
        self.eps = float(eps)

    def forward(self, level_idx: int, base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        weights = F.relu(self.weights[level_idx]).to(device=base.device, dtype=base.dtype)
        return (weights[0] * base + weights[1] * residual) / (weights.sum() + self.eps)


class FullScaleContextAggregation(nn.Module):
    """Aggregate all encoder scales into one context residual per level."""

    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.num_levels = len(in_channels)
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.GroupNorm(1, out_channels),
                    nn.GELU(),
                )
                for channels in in_channels
            ]
        )
        self.fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels * self.num_levels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(1, out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, channels, kernel_size=1),
                )
                for channels in in_channels
            ]
        )

    def forward(self, levels: list[torch.Tensor]) -> list[torch.Tensor]:
        projected = [proj(feat) for proj, feat in zip(self.proj, levels)]
        outputs = []
        for idx, target in enumerate(projected):
            size = target.shape[-2:]
            resized = [
                F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
                if feat.shape[-2:] != size
                else feat
                for feat in projected
            ]
            outputs.append(self.fuse[idx](torch.cat(resized, dim=1)))
        return outputs


class LGDLocalizationHead(nn.Module):
    """LGD head that predicts a 56x56 wall-localization map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 32)
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class LGDFeatureGate(nn.Module):
    """LGD gate driven by the detached 56x56 localization probability."""

    def __init__(self, channels: int, init_gamma: float = 0.0) -> None:
        super().__init__()
        hidden = max(channels // 4, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

    def forward(self, feat: torch.Tensor, localization_probability: torch.Tensor) -> torch.Tensor:
        localization = F.interpolate(
            localization_probability.to(device=feat.device, dtype=feat.dtype),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        gate = self.gate(torch.cat([feat, localization], dim=1))
        support = (0.25 + 0.75 * localization).clamp(0.0, 1.0)
        residual = self.context(feat) * gate * support
        return feat + self.gamma.to(dtype=feat.dtype) * residual


class ModalityContextResidualCalibration(nn.Module):
    """MCRC: lightweight modality-wise context residual calibration."""

    def __init__(self, channels: int, init_gamma: float = 0.0, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.mid = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, groups=1),
        )
        self.high = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, groups=1),
        )
        self.low = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, groups=1),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = self.mid(x)
        high_in = F.max_pool2d(x, kernel_size=2, stride=2)
        low_in = F.avg_pool2d(x, kernel_size=2, stride=2)
        high = F.interpolate(self.high(high_in), size=x.shape[-2:], mode="bilinear", align_corners=False)
        low = F.interpolate(self.low(low_in), size=x.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.mix(mid + high + low)
        return x + self.gamma.to(device=x.device, dtype=x.dtype) * residual


def parse_feature_levels(value: str, num_levels: int) -> set[int]:
    value = str(value).strip().lower()
    if value in {"all", "*"}:
        return set(range(num_levels))
    levels = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        levels.add(int(item))
    return {idx for idx in levels if 0 <= idx < num_levels}


def parse_dilations(value: str) -> list[int]:
    dilations = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            dilations.append(int(item))
    return dilations or [1, 2, 4]


class DilatedSpatialResidualCalibration(nn.Module):
    """DSRC: dilated spatial residual calibration for fused feature maps."""

    def __init__(
        self,
        channels: int,
        init_gamma: float = 0.0,
        dilations: str = "1,2,4",
        reduction: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(channels // int(reduction), 16)
        dilation_values = parse_dilations(dilations)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(1, hidden),
                    nn.GELU(),
                )
                for dilation in dilation_values
            ]
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(hidden * (len(dilation_values) + 1), channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pre(x)
        branch_outputs = [branch(feat) for branch in self.branches]
        global_context = self.global_branch(feat).expand(-1, -1, x.shape[-2], x.shape[-1])
        gate = self.attention(torch.cat(branch_outputs + [global_context], dim=1))
        residual = self.mix(feat * gate)
        return x + self.gamma.to(device=x.device, dtype=x.dtype) * residual


class DualModalFeatureEncoder(nn.Module):
    """Dual encoders followed by DAF and MCE at four feature levels."""

    def __init__(
        self,
        config,
        img_size: int,
        mce_residual_init_weight: float = 0.05,
        fpn_channels: int = 256,
        enable_mcrc: bool = False,
        mcrc_gamma: float = 0.0,
        mcrc_levels: str = "1,2",
        enable_dsrc: bool = False,
        dsrc_gamma: float = 0.02,
        dsrc_levels: str = "1,2",
        dsrc_dilations: str = "1,2,4",
        dsrc_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.n_skip = int(config.n_skip)
        img_size_pair = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (
            img_size_pair[0] // 16 // grid_size[0],
            img_size_pair[1] // 16 // grid_size[1],
        )
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size_pair[0] // patch_size_real[0]) * (img_size_pair[1] // patch_size_real[1])

        self.bmode_encoder = ResNetV2(
            block_units=config.resnet.num_layers,
            width_factor=config.resnet.width_factor,
        )
        self.ceus_encoder = ResNetV2(
            block_units=config.resnet.num_layers,
            width_factor=config.resnet.width_factor,
        )
        deep_channels = self.bmode_encoder.width * 16
        level_channels = [deep_channels] + list(config.skip_channels[: self.n_skip])

        self.daf_blocks = nn.ModuleList([DualModalAdaptiveFusion(ch) for ch in level_channels])
        self.enable_mcrc = bool(enable_mcrc)
        self.mcrc_levels = parse_feature_levels(mcrc_levels, len(level_channels))
        self.bmode_mcrc_blocks = nn.ModuleList(
            [ModalityContextResidualCalibration(ch, init_gamma=mcrc_gamma) for ch in level_channels]
        )
        self.ceus_mcrc_blocks = nn.ModuleList(
            [ModalityContextResidualCalibration(ch, init_gamma=mcrc_gamma) for ch in level_channels]
        )
        self.enable_dsrc = bool(enable_dsrc)
        self.dsrc_levels = parse_feature_levels(dsrc_levels, len(level_channels))
        self.dsrc_blocks = nn.ModuleList(
            [
                DilatedSpatialResidualCalibration(
                    ch,
                    init_gamma=dsrc_gamma,
                    dilations=dsrc_dilations,
                    reduction=dsrc_reduction,
                )
                for ch in level_channels
            ]
        )
        self.mce_context_aggregation = FullScaleContextAggregation(
            level_channels, out_channels=fpn_channels
        )
        self.mce_residual_merges = NormalizedResidualMerge(
            len(level_channels), init_context_weight=mce_residual_init_weight
        )
        self.patch_embeddings = nn.Conv2d(
            in_channels=deep_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def collect_levels(self, deep: torch.Tensor, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [deep] + list(features[: self.n_skip])

    def forward(self, bmode: torch.Tensor, ceus: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        bmode_deep, bmode_features = self.bmode_encoder(bmode)
        ceus_deep, ceus_features = self.ceus_encoder(ceus)
        bmode_levels = self.collect_levels(bmode_deep, bmode_features)
        ceus_levels = self.collect_levels(ceus_deep, ceus_features)
        if self.enable_mcrc:
            bmode_levels = [
                self.bmode_mcrc_blocks[idx](feat) if idx in self.mcrc_levels else feat
                for idx, feat in enumerate(bmode_levels)
            ]
            ceus_levels = [
                self.ceus_mcrc_blocks[idx](feat) if idx in self.mcrc_levels else feat
                for idx, feat in enumerate(ceus_levels)
            ]
        daf_features = [
            fusion_block(b_level, c_level)
            for b_level, c_level, fusion_block in zip(
                bmode_levels,
                ceus_levels,
                self.daf_blocks,
            )
        ]
        context_residuals = self.mce_context_aggregation(daf_features)
        mce_features = [
            self.mce_residual_merges(idx, base, residual)
            for idx, (base, residual) in enumerate(zip(daf_features, context_residuals))
        ]
        if self.enable_dsrc:
            mce_features = [
                self.dsrc_blocks[idx](feat) if idx in self.dsrc_levels else feat
                for idx, feat in enumerate(mce_features)
            ]
        x = self.patch_embeddings(mce_features[0])
        x = x.flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings), mce_features[1:]


class TransformerEncoder(nn.Module):
    def __init__(self, config, vis: bool) -> None:
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([copy.deepcopy(Block(config, vis)) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


class TAKNetTransformer(nn.Module):
    def __init__(
        self,
        config,
        img_size: int,
        vis: bool,
        mce_residual_init_weight: float,
        enable_mcrc: bool = False,
        mcrc_gamma: float = 0.0,
        mcrc_levels: str = "1,2",
        enable_dsrc: bool = False,
        dsrc_gamma: float = 0.02,
        dsrc_levels: str = "1,2",
        dsrc_dilations: str = "1,2,4",
        dsrc_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.embeddings = DualModalFeatureEncoder(
            config,
            img_size=img_size,
            mce_residual_init_weight=mce_residual_init_weight,
            enable_mcrc=enable_mcrc,
            mcrc_gamma=mcrc_gamma,
            mcrc_levels=mcrc_levels,
            enable_dsrc=enable_dsrc,
            dsrc_gamma=dsrc_gamma,
            dsrc_levels=dsrc_levels,
            dsrc_dilations=dsrc_dilations,
            dsrc_reduction=dsrc_reduction,
        )
        self.encoder = TransformerEncoder(config, vis)

    def forward(self, bmode: torch.Tensor, ceus: torch.Tensor):
        embedding_output, features = self.embeddings(bmode, ceus)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class TAKNet(nn.Module):
    """TAK-Net for paired B-mode/CEUS carotid wall segmentation."""

    locator_stage_index = 1
    gate_stage_index = 1
    auxiliary_stage_index = 2

    def __init__(
        self,
        config,
        img_size: int = 224,
        vis: bool = False,
        mce_residual_init_weight: float = 0.05,
        lgd_gate_gamma: float = 0.0,
        enable_locator_head: bool = True,
        enable_locator_gate: bool = True,
        enable_aux_head: bool = True,
        enable_mcrc: bool = False,
        mcrc_gamma: float = 0.0,
        mcrc_levels: str = "1,2",
        enable_dsrc: bool = False,
        dsrc_gamma: float = 0.02,
        dsrc_levels: str = "1,2",
        dsrc_dilations: str = "1,2,4",
        dsrc_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_locator_head = bool(enable_locator_head)
        self.enable_locator_gate = bool(enable_locator_gate)
        self.enable_aux_head = bool(enable_aux_head)
        self.transformer = TAKNetTransformer(
            config,
            img_size,
            vis,
            mce_residual_init_weight,
            enable_mcrc=enable_mcrc,
            mcrc_gamma=mcrc_gamma,
            mcrc_levels=mcrc_levels,
            enable_dsrc=enable_dsrc,
            dsrc_gamma=dsrc_gamma,
            dsrc_levels=dsrc_levels,
            dsrc_dilations=dsrc_dilations,
            dsrc_reduction=dsrc_reduction,
        )
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],
            out_channels=config["n_classes"],
            kernel_size=3,
        )
        self.locator_head = LGDLocalizationHead(config["decoder_channels"][self.locator_stage_index])
        self.locator_gate = LGDFeatureGate(
            config["decoder_channels"][self.gate_stage_index],
            init_gamma=lgd_gate_gamma,
        )
        self.auxiliary_head = nn.Conv2d(
            config["decoder_channels"][self.auxiliary_stage_index], 1, kernel_size=1
        )

    def forward(self, bmode: torch.Tensor, ceus: torch.Tensor, return_aux: bool = False):
        x, _, features = self.transformer(bmode, ceus)
        batch_size, n_patch, hidden = x.size()
        side = int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(batch_size, hidden, side, side)
        x = self.decoder.conv_more(x)

        locator_logits = None
        auxiliary_logits = []
        for idx, decoder_block in enumerate(self.decoder.blocks):
            skip = features[idx] if idx < self.config.n_skip else None
            x = decoder_block(x, skip=skip)
            if idx == self.locator_stage_index and self.enable_locator_head:
                locator_logits = self.locator_head(x)
                if self.enable_locator_gate:
                    localization_probability = torch.sigmoid(locator_logits.float()).to(dtype=x.dtype).detach()
                    x = self.locator_gate(x, localization_probability)
            if idx == self.auxiliary_stage_index and self.enable_aux_head:
                auxiliary_logits.append(self.auxiliary_head(x))

        logits = self.segmentation_head(x)
        if return_aux:
            return {
                "logits": logits,
                "aux_logits": auxiliary_logits,
                "locator_logits": [] if locator_logits is None else [locator_logits],
            }
        return logits

    def load_from(self, weights) -> None:
        embeddings = self.transformer.embeddings
        with torch.no_grad():
            embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                embeddings.position_embeddings.copy_(posemb)
            elif posemb.size(1) - 1 == posemb_new.size(1):
                embeddings.position_embeddings.copy_(posemb[:, 1:])
            else:
                logger.info("load_pretrained: resized position embedding %s to %s", posemb.size(), posemb_new.size())
                ntok_new = posemb_new.size(1)
                posemb_grid = posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                posemb_grid = ndimage.zoom(posemb_grid, (gs_new / gs_old, gs_new / gs_old, 1), order=1)
                embeddings.position_embeddings.copy_(np2th(posemb_grid.reshape(1, gs_new * gs_new, -1)))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for _, block in self.transformer.encoder.named_children():
                if isinstance(block, nn.ModuleList):
                    for unit_name, unit in block.named_children():
                        unit.load_from(weights, n_block=unit_name)

            for encoder in (embeddings.bmode_encoder, embeddings.ceus_encoder):
                encoder.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                encoder.root.gn.weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
                encoder.root.gn.bias.copy_(np2th(weights["gn_root/bias"]).view(-1))
                for block_name, block in encoder.body.named_children():
                    for unit_name, unit in block.named_children():
                        unit.load_from(weights, n_block=block_name, n_unit=unit_name)
