#run.py

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor
from PIL import Image

from attack import pla_attack, llava_generate, PLAConfig
from model_loader import load_llava
from utils import list_images_imagenette, sample_images, pil_load, build_chat_prompt_with_image


def find_model_dirs(root: str) -> List[str]:
    """
    Find candidate model directories under root.
    Accept both HF (config.json) and official LLaVA checkpoints (pytorch_model.bin/mm_projector.bin).
    """
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


def contains_target(output_text: str, target: str) -> bool:
    # strict substring match (you can swap to semantic match later)
    return target.lower() in output_text.lower()


def evaluate_model_on_triggers(
    model_path: str,
    triggers: List[Tuple[str, torch.Tensor, Dict[str, Any]]],
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int,
    model_format: str,
    model_base: str | None,
) -> Dict[str, Any]:
    """
    Evaluate one model on a list of trigger images.
    triggers: list of (image_id, adv_image_01_tensor, info)
    """
    model, processor = load_llava(
        model_path,
        device=device,
        dtype=dtype,
        model_format=model_format,
        model_base=model_base,
    )
    tmr_hits = 0
    outputs_preview = []

    for i, (img_id, adv_x, info) in enumerate(triggers):
        prompt = build_chat_prompt_with_image(processor, info["question"])

        target = info["target"]
        out = llava_generate(
            model=model,
            processor=processor,
            image_tensor_01=adv_x.to(device=device, dtype=torch.float32),
            prompt=prompt,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
        hit = contains_target(out, target)
        tmr_hits += int(hit)

        if i < 3:
            outputs_preview.append({"image_id": img_id, "output": out, "hit": hit})

    tmr = tmr_hits / max(1, len(triggers))
    # cleanup
    del model
    torch.cuda.empty_cache()

    return {"model_path": model_path, "tmr": tmr, "n": len(triggers), "preview": outputs_preview}


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette_dir", type=str, default='/root/autodl-tmp/imagenette/imagenette2/val', help="Path to imagenette2/val (or train)")
    ap.add_argument("--base_model", type=str, default='/root/autodl-tmp/LLaVA-ori', help="Path to base LLaVA checkpoint directory")
    ap.add_argument("--finetuned_root", type=str, default='/root/autodl-tmp/LLaVAs', help="Path to root folder containing finetuned model subdirs")
    ap.add_argument("--question", type=str, default='Detecting copyright.', help="Prompt trigger text (user question)")
    ap.add_argument("--target", type=str, default='CVPR conference', help="Target output text used in attack & evaluation")
    ap.add_argument("--num_images", type=int, default=200, help="Number of Imagenette images to sample (paper uses 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=16/255)
    ap.add_argument("--alpha", type=float, default=1/255)
    ap.add_argument("--model_lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=None, help="Deprecated alias for --model_lr")
    ap.add_argument("--model_grad_clip", type=float, default=5e-3)
    ap.add_argument("--grad_clip", type=float, default=None, help="Deprecated alias for --model_grad_clip")
    ap.add_argument("--reg_lambda", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--out_dir", type=str, default="./pla_runs/run1")
    ap.add_argument("--save_triggers", default=True)
    ap.add_argument("--save_loss_curves", action="store_true", help="Save per-image loss curves to a separate folder")
    ap.add_argument("--model_format", type=str, default="auto", choices=["auto", "hf", "llava"])
    ap.add_argument("--model_base", type=str, default=None, help="Base model path for official LLaVA checkpoints")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    if args.beta is not None and args.model_lr == 1e-4:
        args.model_lr = args.beta
    if args.grad_clip is not None and args.model_grad_clip == 5e-3:
        args.model_grad_clip = args.grad_clip

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trig_dir = out_dir / "triggers"
    trig_dir.mkdir(parents=True, exist_ok=True)
    loss_dir = out_dir / "loss_curves"
    if args.save_loss_curves:
        loss_dir.mkdir(parents=True, exist_ok=True)

    # 1) sample images
    all_imgs = list_images_imagenette(args.imagenette_dir)
    if len(all_imgs) == 0:
        raise RuntimeError(f"No images found under {args.imagenette_dir}")
    sampled = sample_images(all_imgs, args.num_images, seed=args.seed)

    cfg = PLAConfig(
        steps=args.steps,
        epsilon=args.epsilon,
        alpha=args.alpha,
        model_lr=args.model_lr,
        model_grad_clip=args.model_grad_clip,
        reg_lambda=args.reg_lambda,
        seed=args.seed,
    )

    triggers = []
    trigger_meta = []

    triggers_json = out_dir / "triggers.json"

    # If triggers exist, reuse them
    if triggers_json.exists() and (trig_dir.exists()) and (len(list(trig_dir.glob("*.png"))) > 0):
        print(f"[reuse] Found existing triggers under {trig_dir}, loading and skipping attack.")

        trigger_meta = json.loads(triggers_json.read_text(encoding="utf-8"))

        to_tensor = ToTensor()
        for meta in trigger_meta:
            img_id = meta["image_id"]
            png_path = trig_dir / f"{img_id}.png"
            if not png_path.exists():
                raise RuntimeError(f"Missing trigger file: {png_path}")

            # load png -> (1,3,H,W) float in [0,1]
            pil = Image.open(png_path).convert("RGB")
            adv_x = to_tensor(pil).unsqueeze(0)

            # info should be what pla_attack saved (prompt/target included in meta)
            info = {
                "steps": meta.get("steps"),
                "epsilon": meta.get("epsilon"),
                "alpha": meta.get("alpha"),
                "model_lr": meta.get("model_lr", meta.get("beta")),
                "model_grad_clip": meta.get("model_grad_clip", meta.get("grad_clip")),
                "reg_lambda": meta.get("reg_lambda", 0.0),
                "best_loss": meta.get("best_loss"),
                "question": meta.get("question"),
                "target": meta.get("target"),
                "prompt": meta.get("prompt"),
            }

            triggers.append((img_id, adv_x, info))

    else:
        # 2) load base and craft triggers
        base_model, base_processor = load_llava(
            args.base_model,
            device=args.device,
            dtype=dtype,
            model_format=args.model_format,
            model_base=args.model_base,
        )

        # 1) sample images (you already did this above)
        for idx, img_path in enumerate(sampled):
            pil_img = pil_load(img_path)

            adv_x, info = pla_attack(
                model=base_model,
                processor=base_processor,
                pil_image=pil_img,
                question_text=args.question,
                target_text=args.target,
                cfg=cfg,
                record_losses=args.save_loss_curves,
                device=args.device,
                dtype=dtype,
                done=idx,
                todo=len(sampled),
            )

            img_id = f"{idx:04d}"
            triggers.append((img_id, adv_x.cpu(), info))

            if args.save_loss_curves:
                losses = info.pop("losses", None)
                if losses is not None:
                    loss_path = loss_dir / f"{img_id}.json"
                    loss_path.write_text(json.dumps({"loss": losses}, ensure_ascii=False, indent=2), encoding="utf-8")

            meta = {"image_id": img_id, "src_path": img_path, **info}
            trigger_meta.append(meta)

            if args.save_triggers:
                from torchvision.utils import save_image
                save_image(adv_x.cpu(), str(trig_dir / f"{img_id}.png"))

            if (idx + 1) % 10 == 0:
                print(f"[attack] {idx+1}/{len(sampled)} done")

        # cleanup base model to free VRAM before evaluation
        del base_model
        torch.cuda.empty_cache()

        (out_dir / "triggers.json").write_text(json.dumps(trigger_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) evaluate base + finetuned
    model_dirs = [args.base_model] + find_model_dirs(args.finetuned_root)

    results = []
    previews = {}

    for mp in model_dirs:
        print(f"[eval] loading {mp}")
        r = evaluate_model_on_triggers(
            model_path=mp,
            triggers=triggers,
            device=args.device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            model_format=args.model_format,
            model_base=args.model_base,
        )
        results.append({"model": mp, "tmr": r["tmr"], "n": r["n"]})
        previews[mp] = r["preview"]
        print(f"[eval] TMR={r['tmr']:.4f} ({r['n']} samples)")

    # write csv
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "tmr", "n"])
        w.writeheader()
        for row in results:
            w.writerow(row)

    (out_dir / "previews.json").write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
