# attack.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration
from torchvision.transforms.functional import to_pil_image
from transformers.utils.generic import ModelOutput

from utils import (
    preprocess_image_to_tensor,
    normalize_for_llava,
    build_chat_prompt_with_image,
    prepare_text_inputs_and_labels,
)


@dataclass
class PLAConfig:
    steps: int = 1000
    epsilon: float = 16 / 255
    alpha: float = 1 / 255
    model_lr: float = 1e-4
    model_grad_clip: float = 5e-3
    reg_lambda: float = 0.0
    # optional: if you want deterministic
    seed: Optional[int] = 0


def ensure_llava_processor_config(model: LlavaForConditionalGeneration, processor):
    # 1) patch_size 必须来自当前 model
    ps = getattr(getattr(model.config, "vision_config", None), "patch_size", None)
    if ps is not None:
        processor.patch_size = int(ps)

    # 2) vision_feature_select_strategy 必须来自当前 model（不同 ckpt 可能不同）
    vfs = getattr(model.config, "vision_feature_select_strategy", None)
    if vfs is not None:
        processor.vision_feature_select_strategy = vfs
    else:
        # fallback
        processor.vision_feature_select_strategy = getattr(processor, "vision_feature_select_strategy", "default")

    # 3) num_additional_image_tokens：优先读 model.config（有些 ckpt 会存）
    nai = getattr(model.config, "num_additional_image_tokens", None)
    if nai is None:
        # 再退回 processor 自己已有的；否则按 LLaVA-1.5 常见值 0
        nai = getattr(processor, "num_additional_image_tokens", 0)
    processor.num_additional_image_tokens = int(nai)


def _count_image_tokens(processor, input_ids: torch.Tensor) -> int:
    img_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    return int((input_ids == img_id).sum().item())


def _unwrap_to_tensor(x):
    """
    Unwrap common nested returns (list/tuple) to get the last Tensor.
    """
    import torch
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        # take the last element (often the most processed feature)
        return _unwrap_to_tensor(x[-1])
    if isinstance(x, ModelOutput):
        # prefer last_hidden_state (B, L, D)
        if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
            return x.last_hidden_state
        # fallback to pooler_output (B, D)
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        # sometimes models use "hidden_states" etc.
        if hasattr(x, "hidden_states") and x.hidden_states is not None:
            # last layer hidden state
            return x.hidden_states[-1]
        raise TypeError(f"ModelOutput has no usable tensor fields: {x.keys()}")
    raise TypeError(f"Unknown feature return type: {type(x)}")


@torch.no_grad()
def _infer_num_image_features(
    model: LlavaForConditionalGeneration,
    pixel_values: torch.Tensor,
    vision_feature_select_strategy: Optional[str] = None,
) -> int:
    """
    Return number of image feature tokens N.

    IMPORTANT:
    Some checkpoints set model.config.vision_feature_select_strategy="full",
    which can make image_features a 4D spatial map (B, C, H, W). In newer
    transformers versions, the mismatch check may use numel-per-sample,
    producing huge numbers like 1024*48*48 = 2359296.

    So we MUST infer features using the SAME strategy we will use in forward/generate.
    """
    # Preferred: model.get_image_features
    if hasattr(model, "get_image_features"):
        try:
            feats_raw = model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
        except TypeError:
            # older transformers: get_image_features may not accept the kwarg
            feats_raw = model.get_image_features(pixel_values=pixel_values)

        feats = _unwrap_to_tensor(feats_raw)  # should be Tensor now

        # common shapes:
        # (B, N, D)  -> N
        # (N, D)     -> N
        # (B, C, H, W) -> treat as H*W (tokens), NOT C*H*W
        if feats.dim() == 3:
            return int(feats.shape[1])
        if feats.dim() == 2:
            return int(feats.shape[0])
        if feats.dim() == 4:
            # (B, C, H, W) -> tokens likely correspond to spatial positions H*W
            return int(feats.shape[-2] * feats.shape[-1])
        raise ValueError(f"Unexpected image feature tensor shape: {tuple(feats.shape)}")

    # Fallback: vision tower direct
    vt = getattr(model, "vision_tower", None) or getattr(getattr(model, "model", None), "vision_tower", None)
    if vt is None:
        raise RuntimeError("Cannot find vision tower to infer image feature length.")

    outs = vt(pixel_values)
    outs_t = _unwrap_to_tensor(outs)

    if outs_t.dim() == 3:
        return int(outs_t.shape[1])
    if outs_t.dim() == 2:
        return int(outs_t.shape[0])
    if outs_t.dim() == 4:
        return int(outs_t.shape[-2] * outs_t.shape[-1])
    raise ValueError(f"Unexpected vision output shape: {tuple(outs_t.shape)}")


def build_inputs_matched(
    model: LlavaForConditionalGeneration,
    processor,
    img_pil,
    prompt: str,
    device: str,
    dtype: torch.dtype,
):
    # 候选组合：通常只需要 nai=0/1；strategy 常见 default/full
    nai_candidates = [0, 1]
    vfs_candidates = [
        getattr(model.config, "vision_feature_select_strategy", None),
        "default",
        "full",
    ]
    vfs_candidates = [x for x in vfs_candidates if x is not None]

    # 先用 processor 正常生成 pixel_values
    base = processor(images=img_pil, text=prompt, return_tensors="pt")
    pixel_values = base["pixel_values"].to(device=device, dtype=dtype)

    # 用与 forward/generate 一致的 strategy 推断特征 token 数
    #（先用 processor 当前值，后面循环会覆盖）
    n_feat = _infer_num_image_features(
        model, pixel_values, vision_feature_select_strategy=getattr(processor, "vision_feature_select_strategy", None)
    )

    last_err = None
    for nai in nai_candidates:
        for vfs in vfs_candidates:
            processor.num_additional_image_tokens = int(nai)
            processor.vision_feature_select_strategy = vfs

            inp = processor(images=img_pil, text=prompt, return_tensors="pt")
            n_tok = _count_image_tokens(processor, inp["input_ids"])

            # 重新用同一个 vfs 推断 n_feat（关键！）
            pv = inp["pixel_values"].to(device=device, dtype=dtype)
            n_feat_try = _infer_num_image_features(model, pv, vision_feature_select_strategy=vfs)

            if n_tok == n_feat_try:
                # 成功：把 processor 输出的所有张量字段都搬到 device
                out = {}
                for k, v in inp.items():
                    if torch.is_tensor(v):
                        if k == "pixel_values":
                            out[k] = v.to(device=device, dtype=dtype)
                        else:
                            out[k] = v.to(device)
                    else:
                        # e.g. list/tuple metadata
                        out[k] = v

                # 额外：把 strategy 显式传给模型，避免 model.config 使用另一个默认值
                out["vision_feature_select_strategy"] = vfs
                return out

            last_err = (nai, vfs, n_tok, n_feat_try)

    raise ValueError(
        f"Cannot match image tokens/features. last try nai={last_err[0]}, vfs={last_err[1]}, "
        f"tokens={last_err[2]}, features={last_err[3]}"
    )


@torch.no_grad()
def llava_generate(
    model: LlavaForConditionalGeneration,
    processor,
    image_tensor_01: torch.Tensor,   # (1,3,H,W) in [0,1]
    prompt: str,
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int = 200,
) -> str:
    ensure_llava_processor_config(model, processor)
    # prompt 和图像都交给同一个 processor 来产出 matched inputs
    img_pil = to_pil_image(image_tensor_01.squeeze(0).detach().cpu())

    proc_inputs = build_inputs_matched(
        model=model,
        processor=processor,
        img_pil=img_pil,
        prompt=prompt,
        device=device,
        dtype=dtype,
    )

    # 显式传入 vision_feature_select_strategy（已在 proc_inputs 里写好）
    out = model.generate(
        **proc_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return processor.decode(out[0], skip_special_tokens=True).strip()


def pla_attack(
    model: LlavaForConditionalGeneration,
    processor,
    pil_image,
    done,
    todo,
    question_text: str,
    target_text: str,
    cfg: PLAConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run PLA-style optimization to produce a trigger image.

    Returns:
        adv_image_01: torch.Tensor in [0,1], shape (1,3,H,W) at model resolution
        info: dict containing metadata
    """
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    ensure_llava_processor_config(model, processor)

    prompt = build_chat_prompt_with_image(processor, question_text)

    # Preprocess the PIL image to tensor [0,1] at model resolution (differentiable steps are after this).
    x0 = preprocess_image_to_tensor(processor, pil_image).to(device=device, dtype=torch.float32)  # keep float32 for stability
    x = x0.clone().detach()

    # Prepare text inputs and labels for teacher-forcing loss on target.
    text_inputs, labels = prepare_text_inputs_and_labels(
        processor, pil_image, prompt, target_text, device=device
    )

    best_loss = float("inf")
    best_x = x.clone().detach()

    # 用 processor 当前 strategy；避免 model.config 默认值（某些 ckpt 是 "full"）导致 forward mismatch
    vfs = getattr(processor, "vision_feature_select_strategy", None)

    for step in tqdm(range(cfg.steps)):
        x.requires_grad_(True)

        pixel_values = normalize_for_llava(processor, x).to(device=device, dtype=dtype)

        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask", None),
            pixel_values=pixel_values,
            labels=labels,
            vision_feature_select_strategy=vfs,
        )
        loss = outputs.loss

        if cfg.reg_lambda > 0:
            reg = (x - x0).pow(2).mean()
            total = loss + cfg.reg_lambda * reg
        else:
            total = loss

        total.backward()

        with torch.no_grad():
            # Update model parameters (PLA) with gradient ascent on loss
            if cfg.model_lr and cfg.model_lr > 0:
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad
                    if cfg.model_grad_clip and cfg.model_grad_clip > 0:
                        g = g.clamp(min=-cfg.model_grad_clip, max=cfg.model_grad_clip)
                    p.add_(cfg.model_lr * g)

            # Update image with FGSM-style sign step
            if x.grad is not None:
                x = x - cfg.alpha * x.grad.sign()
            x = torch.max(torch.min(x, x0 + cfg.epsilon), x0 - cfg.epsilon)
            x = x.clamp(0.0, 1.0)

        cur_loss = float(loss.detach().cpu())
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_x = x.clone().detach()

        print(f"{done}/{todo} images done | Current loss:{cur_loss} | Best loss:{best_loss}")

        x = x.detach()

    info = {
        "steps": cfg.steps,
        "epsilon": cfg.epsilon,
        "alpha": cfg.alpha,
        "model_lr": cfg.model_lr,
        "model_grad_clip": cfg.model_grad_clip,
        "reg_lambda": cfg.reg_lambda,
        "best_loss": best_loss,
        "question": question_text,
        "target": target_text,
        "prompt": prompt,
    }
    return best_x.to(dtype=torch.float32), info
