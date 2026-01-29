from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

from attack import PLAConfig, pla_attack
from model_loader import load_llava
from utils import list_images_imagenette, sample_images, pil_load, build_chat_prompt_with_image


def find_model_dirs(root: str) -> List[str]:
    root_p = Path(root)
    out = []
    for p in sorted(root_p.iterdir()):
        if not p.is_dir():
            continue
        if (p / "config.json").exists():
            out.append(str(p))
            continue
        if (p / "pytorch_model.bin").exists() or (p / "mm_projector.bin").exists():
            out.append(str(p))
            continue
        if any(p.glob("model*.safetensors")):
            out.append(str(p))
    return out


def _tensor_stats(diff: torch.Tensor, base: torch.Tensor, compute_rank: bool, rank_max_dim: int, rank_max_numel: int) -> Dict[str, Any]:
    diff_f = diff.to(dtype=torch.float32)
    base_f = base.to(dtype=torch.float32)
    numel = diff_f.numel()

    sum_ = diff_f.sum().item()
    sumsq = (diff_f * diff_f).sum().item()
    sumabs = diff_f.abs().sum().item()
    maxabs = diff_f.abs().max().item() if numel > 0 else 0.0

    mean = sum_ / max(1, numel)
    var = max(0.0, (sumsq / max(1, numel)) - (mean * mean))
    std = var ** 0.5

    base_norm = (base_f * base_f).sum().sqrt().item()
    l2 = (sumsq ** 0.5) if sumsq > 0 else 0.0
    l2_rel = l2 / (base_norm + 1e-12)

    rank = None
    if compute_rank and diff_f.dim() >= 2:
        rows = diff_f.shape[0]
        cols = diff_f.numel() // rows
        if rows <= rank_max_dim and cols <= rank_max_dim and diff_f.numel() <= rank_max_numel:
            diff_2d = diff_f.reshape(rows, cols)
            rank = int(torch.linalg.matrix_rank(diff_2d).item())

    return {
        "numel": int(numel),
        "shape": list(diff.shape),
        "l2": float(l2),
        "l2_rel": float(l2_rel),
        "linf": float(maxabs),
        "mean": float(mean),
        "std": float(std),
        "mean_abs": float(sumabs / max(1, numel)),
        "rank": rank,
    }


def compute_drift_stats(
    base_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    compute_rank: bool,
    rank_max_dim: int,
    rank_max_numel: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base_keys = set(base_state.keys())
    model_keys = set(model_state.keys())
    common = sorted(base_keys & model_keys)

    missing_in_model = sorted(base_keys - model_keys)
    missing_in_base = sorted(model_keys - base_keys)

    layers: Dict[str, Any] = {}
    total_numel = 0
    total_sum = 0.0
    total_sumsq = 0.0
    total_sumabs = 0.0
    total_maxabs = 0.0
    total_l2sq = 0.0

    for name in common:
        b = base_state[name]
        w = model_state[name].detach().cpu()
        if b.shape != w.shape:
            layers[name] = {
                "shape_base": list(b.shape),
                "shape_model": list(w.shape),
                "mismatch": True,
            }
            continue

        diff = w - b
        st = _tensor_stats(diff, b, compute_rank, rank_max_dim, rank_max_numel)
        layers[name] = st

        numel = st["numel"]
        total_numel += numel
        total_sum += st["mean"] * numel
        total_sumabs += st["mean_abs"] * numel
        total_maxabs = max(total_maxabs, st["linf"])
        total_l2sq += st["l2"] ** 2
        total_sumsq += (st["std"] ** 2 + st["mean"] ** 2) * numel

    mean = total_sum / max(1, total_numel)
    var = max(0.0, (total_sumsq / max(1, total_numel)) - (mean * mean))
    std = var ** 0.5

    global_stats = {
        "numel": int(total_numel),
        "l2": float(total_l2sq ** 0.5),
        "linf": float(total_maxabs),
        "mean": float(mean),
        "std": float(std),
        "mean_abs": float(total_sumabs / max(1, total_numel)),
        "matched_keys": len(common),
        "missing_in_model": len(missing_in_model),
        "missing_in_base": len(missing_in_base),
        "missing_in_model_examples": missing_in_model[:50],
        "missing_in_base_examples": missing_in_base[:50],
    }

    return global_stats, layers


def choose_image(imagenette_dir: str, seed: int, image_path: str | None, index: int) -> str:
    if image_path:
        return image_path
    all_imgs = list_images_imagenette(imagenette_dir)
    if not all_imgs:
        raise RuntimeError(f"No images found under {imagenette_dir}")
    sampled = sample_images(all_imgs, max(index + 1, 1), seed=seed)
    return sampled[index]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette_dir", type=str, default="/root/autodl-tmp/imagenette/imagenette2/val")
    ap.add_argument("--image_path", type=str, default=None, help="Optional explicit image path for PLA")
    ap.add_argument("--image_index", type=int, default=0)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--finetuned_root", type=str, default=None)
    ap.add_argument("--question", type=str, default="Detecting copyright.")
    ap.add_argument("--target", type=str, default="CVPR conference")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=16 / 255)
    ap.add_argument("--alpha", type=float, default=1 / 255)
    ap.add_argument("--model_lr", type=float, default=1e-4)
    ap.add_argument("--model_grad_clip", type=float, default=5e-3)
    ap.add_argument("--reg_lambda", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--model_format", type=str, default="auto", choices=["auto", "hf", "llava"])
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="./investigate_runs/run1")
    ap.add_argument("--no_pla", dest="run_pla", action="store_false")
    ap.add_argument("--compute_rank", action="store_true", default=True)
    ap.add_argument("--no_rank", dest="compute_rank", action="store_false")
    ap.add_argument("--rank_max_dim", type=int, default=4096)
    ap.add_argument("--rank_max_numel", type=int, default=20_000_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drift_dir = out_dir / "drifts"
    drift_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    base_model, base_processor = load_llava(
        args.base_model,
        device=args.device,
        dtype=dtype,
        model_format=args.model_format,
        model_base=args.model_base,
    )
    base_state = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

    meta_common = {
        "base_model": args.base_model,
        "model_format": args.model_format,
        "model_base": args.model_base,
        "dtype": args.dtype,
    }

    if args.run_pla:
        img_path = choose_image(args.imagenette_dir, args.seed, args.image_path, args.image_index)
        pil_img = pil_load(img_path)

        cfg = PLAConfig(
            steps=args.steps,
            epsilon=args.epsilon,
            alpha=args.alpha,
            model_lr=args.model_lr,
            model_grad_clip=args.model_grad_clip,
            reg_lambda=args.reg_lambda,
            seed=args.seed,
        )

        adv_x, info = pla_attack(
            model=base_model,
            processor=base_processor,
            pil_image=pil_img,
            question_text=args.question,
            target_text=args.target,
            cfg=cfg,
            record_losses=False,
            device=args.device,
            dtype=dtype,
            done=0,
            todo=1,
        )
        _ = adv_x

        global_stats, layers = compute_drift_stats(
            base_state,
            base_model.state_dict(),
            args.compute_rank,
            args.rank_max_dim,
            args.rank_max_numel,
        )

        payload = {
            "meta": {
                **meta_common,
                "mode": "pla",
                "image_path": img_path,
                "question": args.question,
                "target": args.target,
                "attack_config": info,
            },
            "global": global_stats,
            "layers": layers,
        }
        (drift_dir / "pla_drift.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.finetuned_root:
        fin_dir = drift_dir / "finetuned"
        fin_dir.mkdir(parents=True, exist_ok=True)

        model_dirs = find_model_dirs(args.finetuned_root)
        for mp in model_dirs:
            model, _ = load_llava(
                mp,
                device=args.device,
                dtype=dtype,
                model_format=args.model_format,
                model_base=args.model_base,
            )

            global_stats, layers = compute_drift_stats(
                base_state,
                model.state_dict(),
                args.compute_rank,
                args.rank_max_dim,
                args.rank_max_numel,
            )

            payload = {
                "meta": {
                    **meta_common,
                    "mode": "finetuned",
                    "model_path": mp,
                },
                "global": global_stats,
                "layers": layers,
            }
            name = Path(mp).name
            (fin_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            del model
            torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
