from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def _is_official_llava_model(model_path: str) -> bool:
    name = Path(model_path).name.lower()
    return ("llava-medf" in name) or ("llava-medif" in name)


def _filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class LLaVAProcessorAdapter:
    """
    Minimal adapter to look like a HF LlavaProcessor, backed by LLaVA official tokenizer + image processor.
    """

    def __init__(self, tokenizer, image_processor, conv_mode: str, mm_use_im_start_end: bool):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end
        # allow patching by ensure_llava_processor_config
        self.patch_size = getattr(getattr(image_processor, "config", None), "patch_size", None)
        self.vision_feature_select_strategy = getattr(tokenizer, "vision_feature_select_strategy", None)
        self.num_additional_image_tokens = 0

        try:
            from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        except Exception:
            DEFAULT_IMAGE_TOKEN = "<image>"
            DEFAULT_IM_START_TOKEN = "<im_start>"
            DEFAULT_IM_END_TOKEN = "<im_end>"

        self._image_token = DEFAULT_IMAGE_TOKEN
        self._im_start_token = DEFAULT_IM_START_TOKEN
        self._im_end_token = DEFAULT_IM_END_TOKEN

        try:
            from llava.conversation import conv_templates
            self._conv_templates = conv_templates
        except Exception:
            self._conv_templates = None

    def apply_chat_template(self, conversation, add_generation_prompt: bool = True) -> str:
        if not self._conv_templates:
            raise RuntimeError("LLaVA conversation templates not available; install the official LLaVA package.")
        if not conversation:
            return ""

        user_text = ""
        for part in conversation[0].get("content", []):
            if part.get("type") == "text":
                user_text = part.get("text", "")
                break

        if self.mm_use_im_start_end:
            image_token = f"{self._im_start_token}{self._image_token}{self._im_end_token}"
        else:
            image_token = self._image_token

        qs = image_token + "\n" + user_text

        conv = self._conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        if add_generation_prompt:
            conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def __call__(self, images=None, text=None, return_tensors: Optional[str] = None, add_special_tokens: bool = True):
        out: Dict[str, Any] = {}

        if images is not None:
            image_out = self.image_processor.preprocess(images, return_tensors=return_tensors or "pt")
            if isinstance(image_out, dict):
                out["pixel_values"] = image_out.get("pixel_values", image_out)
            else:
                out["pixel_values"] = image_out

        if text is not None:
            tok = self.tokenizer(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens)
            out.update(tok)

        return out

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


class LLaVAModelAdapter(torch.nn.Module):
    """
    Adapter to normalize the forward/generate signature to use pixel_values.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, *args, **kwargs):
        if "pixel_values" in kwargs and "images" not in kwargs:
            kwargs["images"] = kwargs.pop("pixel_values")
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if "pixel_values" in kwargs and "images" not in kwargs:
            kwargs["images"] = kwargs.pop("pixel_values")
        return self._model.generate(*args, **kwargs)

    def __getattr__(self, item):
        if item == "_model":
            return super().__getattr__(item)
        return getattr(self._model, item)


def load_llava_hf(model_path: str, device: str, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def _infer_conv_mode(model_name: str) -> str:
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mpt" in name:
        return "mpt"
    if "v1" in name:
        return "llava_v1"
    return "llava_v0"


def load_llava_official(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    model_base: Optional[str] = None,
    load_8bit: bool = False,
    load_4bit: bool = False,
):
    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
    except Exception as e:
        raise RuntimeError("Official LLaVA loader not available; install the LLaVA package.") from e

    model_name = get_model_name_from_path(model_path)

    kwargs = _filter_kwargs(
        load_pretrained_model,
        {
            "model_path": model_path,
            "model_base": model_base,
            "model_name": model_name,
            "load_8bit": load_8bit,
            "load_4bit": load_4bit,
            "device_map": "auto" if device == "cuda" else device,
            "device": device,
        },
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(**kwargs)

    if not load_4bit and not load_8bit:
        model.to(dtype=dtype)
    model.eval()

    conv_mode = _infer_conv_mode(model_name)
    mm_use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))
    processor = LLaVAProcessorAdapter(tokenizer, image_processor, conv_mode, mm_use_im_start_end)
    return LLaVAModelAdapter(model), processor


def load_llava(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    model_format: str = "auto",
    model_base: Optional[str] = None,
):
    """
    Load a LLaVA model from either HF or official LLaVA checkpoints.
    Auto mode treats only LLaVA-MedF and LLaVA-MedIf as official LLaVA.
    """
    model_format = model_format.lower()
    if model_format not in {"auto", "hf", "llava"}:
        raise ValueError(f"Unknown model_format: {model_format}")

    if model_format == "auto":
        if _is_official_llava_model(model_path):
            return load_llava_official(
                model_path,
                device=device,
                dtype=dtype,
                model_base=model_base,
            )
        return load_llava_hf(model_path, device=device, dtype=dtype)

    if model_format == "hf":
        return load_llava_hf(model_path, device=device, dtype=dtype)

    return load_llava_official(
        model_path,
        device=device,
        dtype=dtype,
        model_base=model_base,
    )

    raise RuntimeError(f"Failed to load model at {model_path} with format {model_format}")
