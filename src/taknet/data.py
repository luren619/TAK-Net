from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from .config import ExperimentConfig


def augmentation_config() -> dict:
    return {
        "hflip_prob": 0.5,
        "affine_prob": 0.8,
        "rotation_degrees": 15,
        "translate_fraction": 0.06,
        "scale_range": [0.85, 1.15],
        "shear_degrees": 0,
        "intensity_augmentation_independent_per_modality": True,
        "intensity_prob": 0.5,
        "brightness_range": [0.85, 1.15],
        "contrast_range": [0.85, 1.15],
        "gamma_prob": 0.3,
        "gamma_range": [0.85, 1.2],
        "noise_prob": 0.25,
        "gaussian_noise_std": 0.025,
        "speckle_prob": 0.15,
        "speckle_noise_std": 0.03,
        "blur_prob": 0.15,
        "blur_radius_range": [0.2, 0.8],
    }


def read_file_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_sample_path(dataset_root: Path, modality: str, file_name: str) -> Path:
    return dataset_root / modality / file_name


def pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def mask_to_label(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    label = np.zeros(array.shape, dtype=np.int64)
    label[array == 255] = 1
    label[array == 128] = 255
    return torch.from_numpy(label)


def normalize_to_minus_one_one(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - 0.5) / 0.5


def repeat_rgb(gray_tensor: torch.Tensor) -> torch.Tensor:
    return gray_tensor.repeat(3, 1, 1)


def apply_intensity_augmentation(image: Image.Image, cfg: dict) -> Image.Image:
    if random.random() < float(cfg["intensity_prob"]):
        image = ImageEnhance.Brightness(image).enhance(random.uniform(*cfg["brightness_range"]))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(*cfg["contrast_range"]))

    if random.random() < float(cfg["gamma_prob"]):
        gamma = random.uniform(*cfg["gamma_range"])
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        array = np.power(np.clip(array, 0.0, 1.0), gamma)
        image = Image.fromarray(np.uint8(np.clip(array * 255.0, 0, 255)), mode="L")

    if random.random() < float(cfg["noise_prob"]):
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        array = array + np.random.normal(0.0, float(cfg["gaussian_noise_std"]), size=array.shape)
        image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="L")

    if random.random() < float(cfg["speckle_prob"]):
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        noise = np.random.normal(0.0, float(cfg["speckle_noise_std"]), size=array.shape)
        image = Image.fromarray(np.uint8(np.clip(array + array * noise, 0.0, 1.0) * 255.0), mode="L")

    if random.random() < float(cfg["blur_prob"]):
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(*cfg["blur_radius_range"])))

    return image


class CarotidDualModalDataset(Dataset):
    """Paired B-mode/CEUS dataset for carotid wall segmentation."""

    def __init__(
        self,
        dataset_root: str | Path,
        file_list: str | Path | Iterable[str],
        img_size: int,
        augment: bool,
        aug_cfg: dict | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.aug_cfg = aug_cfg or augmentation_config()
        if isinstance(file_list, (str, Path)):
            self.file_names = read_file_list(Path(file_list))
        else:
            self.file_names = [str(name) for name in file_list]
        missing = []
        for name in self.file_names:
            for modality in ("imgs", "ceus", "masks"):
                path = resolve_sample_path(self.dataset_root, modality, name)
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(f"Missing paired files ({len(missing)} total):\n{preview}")

    def __len__(self) -> int:
        return len(self.file_names)

    def apply_geometric_augmentation(
        self,
        bmode: Image.Image,
        ceus: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        if not self.augment:
            return bmode, ceus, mask

        if random.random() < float(self.aug_cfg["hflip_prob"]):
            bmode = TF.hflip(bmode)
            ceus = TF.hflip(ceus)
            mask = TF.hflip(mask)

        if random.random() < float(self.aug_cfg["affine_prob"]):
            angle = random.uniform(-float(self.aug_cfg["rotation_degrees"]), float(self.aug_cfg["rotation_degrees"]))
            max_shift = int(round(float(self.aug_cfg["translate_fraction"]) * self.img_size))
            translations = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
            scale = random.uniform(float(self.aug_cfg["scale_range"][0]), float(self.aug_cfg["scale_range"][1]))
            shear_limit = float(self.aug_cfg["shear_degrees"])
            shear = random.uniform(-shear_limit, shear_limit) if shear_limit else 0.0
            bmode = TF.affine(bmode, angle, translations, scale, [shear, 0.0], InterpolationMode.BILINEAR, fill=0)
            ceus = TF.affine(ceus, angle, translations, scale, [shear, 0.0], InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle, translations, scale, [shear, 0.0], InterpolationMode.NEAREST, fill=0)
        return bmode, ceus, mask

    def __getitem__(self, index: int) -> dict:
        name = self.file_names[index]
        bmode_original = Image.open(resolve_sample_path(self.dataset_root, "imgs", name)).convert("L")
        ceus_original = Image.open(resolve_sample_path(self.dataset_root, "ceus", name)).convert("L")
        mask_original = Image.open(resolve_sample_path(self.dataset_root, "masks", name)).convert("L")
        original_size = bmode_original.size[::-1]

        resize_shape = (self.img_size, self.img_size)
        bmode = bmode_original.resize(resize_shape, Image.BILINEAR)
        ceus = ceus_original.resize(resize_shape, Image.BILINEAR)
        mask = mask_original.resize(resize_shape, Image.NEAREST)

        bmode, ceus, mask = self.apply_geometric_augmentation(bmode, ceus, mask)
        if self.augment and bool(self.aug_cfg["intensity_augmentation_independent_per_modality"]):
            bmode = apply_intensity_augmentation(bmode, self.aug_cfg)
            ceus = apply_intensity_augmentation(ceus, self.aug_cfg)

        bmode_one = pil_to_unit_tensor(bmode)
        ceus_one = pil_to_unit_tensor(ceus)
        return {
            "bmode": normalize_to_minus_one_one(repeat_rgb(bmode_one)),
            "ceus": normalize_to_minus_one_one(repeat_rgb(ceus_one)),
            "label": mask_to_label(mask),
            "file_name": name,
            "original_size": torch.tensor(original_size, dtype=torch.long),
        }


def make_loader(
    cfg: ExperimentConfig,
    file_list: Path,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = CarotidDualModalDataset(
        dataset_root=cfg.dataset_root,
        file_list=file_list,
        img_size=cfg.img_size,
        augment=augment,
        aug_cfg=augmentation_config(),
    )

    def worker_init(worker_id: int) -> None:
        worker_seed = cfg.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
    )


def tensor_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
