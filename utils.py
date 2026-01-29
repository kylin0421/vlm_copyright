#utils.py

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def list_images_imagenette(root: str) -> List[str]:
    """
    Recursively list common image files under Imagenette root (e.g., imagenette2/val).
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root_p = Path(root)
    paths = []
    for p in root_p.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    return paths


def sample_images(paths: List[str], n: int, seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    if n >= len(paths):
        return paths
    return rng.sample(paths, n)


def build_chat_prompt_with_image(processor, user_text: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)



def _get_llava_image_size(processor) -> int:
    """
    Try to infer the model image size from processor.image_processor.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        # reasonable default for llava-1.5
        return 336
    size = getattr(ip, "size", None)
    # CLIPImageProcessor uses dict like {"shortest_edge": 336} or {"height": 336, "width": 336}
    if isinstance(size, dict):
        if "shortest_edge" in size:
            return int(size["shortest_edge"])
        if "height" in size:
            return int(size["height"])
        if "width" in size:
            return int(size["width"])
    if isinstance(size, int):
        return int(size)
    # fallback
    return 336


def preprocess_image_to_tensor(processor, pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a tensor in [0,1] resized/cropped to LLaVA's expected resolution.

    We implement a CLIP-like resize+center-crop pipeline in a differentiable-friendly way.
    """
    size = _get_llava_image_size(processor)
    # Most LLaVA-1.5 uses square crop (336). We'll do:
    # Resize shortest edge to size, then center crop to (size,size).
    # torchvision's Resize when given int -> resize shorter side to int, keep aspect ratio.
    t = transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # -> [0,1], shape (3,H,W)
        ]
    )
    x = t(pil_image.convert("RGB")).unsqueeze(0)  # (1,3,H,W)
    return x


def normalize_for_llava(processor, image_tensor_01: torch.Tensor) -> torch.Tensor:
    """
    Normalize a [0,1] image tensor to LLaVA pixel_values using the processor's mean/std.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        mean = getattr(ip, "image_mean", None)
        std = getattr(ip, "image_std", None)
    else:
        mean = None
        std = None

    # Defaults for CLIP
    if mean is None:
        mean = [0.48145466, 0.4578275, 0.40821073]
    if std is None:
        std = [0.26862954, 0.26130258, 0.27577711]

    mean_t = torch.tensor(mean, device=image_tensor_01.device, dtype=image_tensor_01.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image_tensor_01.device, dtype=image_tensor_01.dtype).view(1, 3, 1, 1)
    return (image_tensor_01 - mean_t) / std_t


def prepare_text_inputs_and_labels(processor, pil_image, prompt: str, target_text: str, device: str = "cuda"):
    """
    IMPORTANT: must call processor(images=..., text=...) so <image> expands to many image tokens.
    prompt: chat template string that includes the image placeholder
    """
    # prompt-only
    prompt_inputs = processor(images=pil_image, text=prompt, return_tensors="pt", add_special_tokens=False)
    # full = prompt + target
    full_text = prompt + target_text
    full_inputs = processor(images=pil_image, text=full_text, return_tensors="pt", add_special_tokens=False)

    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items() if k in ["input_ids", "attention_mask"]}
    full_inputs = {k: v.to(device) for k, v in full_inputs.items() if k in ["input_ids", "attention_mask"]}

    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs.get("attention_mask", None)

    labels = input_ids.clone()
    pl = prompt_inputs["input_ids"].shape[1]
    labels[:, :pl] = -100

    text_inputs = {"input_ids": input_ids}
    if attention_mask is not None:
        text_inputs["attention_mask"] = attention_mask
    return text_inputs, labels




def pil_load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")



