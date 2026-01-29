from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable

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


def choose_images(
    imagenette_dir: str,
    seed: int,
    image_path: str | None,
    index: int,
    count: int,
) -> List[str]:
    if image_path:
        return [image_path] * count
    all_imgs = list_images_imagenette(imagenette_dir)
    if not all_imgs:
        raise RuntimeError(f"No images found under {imagenette_dir}")
    needed = max(index + count, 1)
    sampled = sample_images(all_imgs, needed, seed=seed)
    return sampled[index : index + count]


def _numeric_keys(d: Dict[str, Any]) -> List[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out.append(k)
    return out


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, var ** 0.5


def aggregate_global_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not stats_list:
        return {}
    keys = _numeric_keys(stats_list[0])
    out = {}
    for k in keys:
        vals = [float(s.get(k, 0.0)) for s in stats_list]
        mean, std = _mean_std(vals)
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = std
        out[f"{k}_min"] = min(vals)
        out[f"{k}_max"] = max(vals)
    return out


def aggregate_layer_stats(layers_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not layers_list:
        return {}
    common = set(layers_list[0].keys())
    for l in layers_list[1:]:
        common &= set(l.keys())

    out: Dict[str, Any] = {}
    for name in sorted(common):
        per = [l[name] for l in layers_list]
        if any(p.get("mismatch") for p in per):
            out[name] = {"mismatch": True}
            continue
        keys = [k for k in _numeric_keys(per[0]) if k != "numel"]
        stats = {
            "shape": per[0].get("shape"),
            "numel": per[0].get("numel"),
        }
        for k in keys:
            vals = [float(p.get(k, 0.0)) for p in per if p.get(k) is not None]
            mean, std = _mean_std(vals)
            stats[f"{k}_mean"] = mean
            stats[f"{k}_std"] = std
            stats[f"{k}_min"] = min(vals) if vals else 0.0
            stats[f"{k}_max"] = max(vals) if vals else 0.0
        out[name] = stats
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette_dir", type=str, default="/root/autodl-tmp/imagenette/imagenette2/val")
    ap.add_argument("--image_path", type=str, default=None, help="Optional explicit image path for PLA")
    ap.add_argument("--image_index", type=int, default=0)
    ap.add_argument("--pla_runs", type=int, default=1)
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
        img_paths = choose_images(
            args.imagenette_dir,
            args.seed,
            args.image_path,
            args.image_index,
            args.pla_runs,
        )

        cfg = PLAConfig(
            steps=args.steps,
            epsilon=args.epsilon,
            alpha=args.alpha,
            model_lr=args.model_lr,
            model_grad_clip=args.model_grad_clip,
            reg_lambda=args.reg_lambda,
            seed=args.seed,
        )

        pla_dir = drift_dir / "pla_runs"
        pla_dir.mkdir(parents=True, exist_ok=True)

        pla_globals = []
        pla_layers = []
        pla_files = []

        for i, img_path in enumerate(img_paths):
            with torch.no_grad():
                base_model.load_state_dict(base_state, strict=True)

            pil_img = pil_load(img_path)

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
                done=i,
                todo=len(img_paths),
            )
            _ = adv_x

            global_stats, layers = compute_drift_stats(
                base_state,
                base_model.state_dict(),
                args.compute_rank,
                args.rank_max_dim,
                args.rank_max_numel,
            )
            pla_globals.append(global_stats)
            pla_layers.append(layers)

            payload = {
                "meta": {
                    **meta_common,
                    "mode": "pla",
                    "run_index": i,
                    "image_path": img_path,
                    "question": args.question,
                    "target": args.target,
                    "attack_config": info,
                },
                "global": global_stats,
                "layers": layers,
            }
            out_path = pla_dir / f"pla_{i:03d}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            pla_files.append(str(out_path))

        summary = {
            "meta": {
                **meta_common,
                "mode": "pla",
                "runs": len(img_paths),
                "image_paths": img_paths,
                "question": args.question,
                "target": args.target,
                "attack_config": {
                    "steps": cfg.steps,
                    "epsilon": cfg.epsilon,
                    "alpha": cfg.alpha,
                    "model_lr": cfg.model_lr,
                    "model_grad_clip": cfg.model_grad_clip,
                    "reg_lambda": cfg.reg_lambda,
                },
                "per_run_files": pla_files,
            },
            "global_agg": aggregate_global_stats(pla_globals),
            "layers_agg": aggregate_layer_stats(pla_layers),
        }
        (drift_dir / "pla_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
