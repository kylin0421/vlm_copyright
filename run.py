#run.py

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

import torch
from torchvision.transforms import ToTensor
from PIL import Image

from attack import pla_attack, pla_attack_batch, llava_generate, PLAConfig
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
    ap.add_argument("--num_images", type=int, default=100, help="Number of Imagenette images to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=16/255)
    ap.add_argument("--alpha", type=float, default=1/255)
    ap.add_argument("--model_lr", type=float, default=1e-4)
    ap.add_argument("--model_grad_clip", type=float, default=5e-3)
    ap.add_argument("--reg_lambda", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--out_dir", type=str, default="./pla_runs/run1")
    ap.add_argument("--save_triggers", default=True)
    ap.add_argument("--save_loss_curves", action="store_true", default=True, help="Save per-image loss curves to a separate folder")
    ap.add_argument("--no_save_loss_curves", dest="save_loss_curves", action="store_false", help="Disable loss curves saving")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--model_format", type=str, default="auto", choices=["auto", "hf", "llava"])
    ap.add_argument("--model_base", type=str, default=None, help="Base model path for official LLaVA checkpoints")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

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

    existing_pngs = sorted(trig_dir.glob("*.png"))
    existing_ids = []
    for p in existing_pngs:
        try:
            existing_ids.append(int(p.stem))
        except ValueError:
            continue
    existing_ids = sorted(set(existing_ids))

    meta_by_id: Dict[str, Dict[str, Any]] = {}
    if triggers_json.exists():
        trigger_meta = json.loads(triggers_json.read_text(encoding="utf-8"))
        meta_by_id = {m.get("image_id"): m for m in trigger_meta if m.get("image_id") is not None}
    elif existing_ids:
        print(f"[warn] Found triggers under {trig_dir} but no triggers.json; metadata will be partial.")

    if existing_ids:
        print(f"[reuse] Found {len(existing_ids)} existing triggers under {trig_dir}.")
        to_tensor = ToTensor()
        for idx in existing_ids:
            if idx >= args.num_images:
                continue
            img_id = f"{idx:04d}"
            png_path = trig_dir / f"{img_id}.png"
            if not png_path.exists():
                continue

            pil = Image.open(png_path).convert("RGB")
            adv_x = to_tensor(pil).unsqueeze(0)

            meta = meta_by_id.get(img_id, {})
            info = {
                "steps": meta.get("steps"),
                "epsilon": meta.get("epsilon"),
                "alpha": meta.get("alpha"),
                "model_lr": meta.get("model_lr"),
                "model_grad_clip": meta.get("model_grad_clip"),
                "reg_lambda": meta.get("reg_lambda", 0.0),
                "best_loss": meta.get("best_loss"),
                "question": meta.get("question", args.question),
                "target": meta.get("target", args.target),
                "prompt": meta.get("prompt"),
            }

            triggers.append((img_id, adv_x, info))

        if len(existing_ids) >= args.num_images:
            print(f"[reuse] Already have {len(existing_ids)} triggers, target is {args.num_images}.")

    if len(triggers) < args.num_images:
        # 2) load base and craft triggers
        base_model, base_processor = load_llava(
            args.base_model,
            device=args.device,
            dtype=dtype,
            model_format=args.model_format,
            model_base=args.model_base,
        )
        base_state = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

        missing_indices = [i for i in range(args.num_images) if i not in existing_ids]
        if not missing_indices:
            print("[attack] No missing triggers to generate.")

        if args.batch_size <= 1:
            for idx in missing_indices:
                # Reset model to initial weights for each image (matches PLA setup)
                with torch.no_grad():
                    base_model.load_state_dict(base_state, strict=True)

                img_path = sampled[idx]
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
                    todo=args.num_images,
                )

                img_id = f"{idx:04d}"
                triggers.append((img_id, adv_x.cpu(), info))

                if args.save_loss_curves:
                    losses = info.pop("losses", None)
                    if losses is not None:
                        loss_path = loss_dir / f"{img_id}.json"
                        loss_path.write_text(json.dumps({"loss": losses}, ensure_ascii=False, indent=2), encoding="utf-8")

                meta = {"image_id": img_id, "src_path": img_path, **info}
                meta_by_id[img_id] = meta

                if args.save_triggers:
                    from torchvision.utils import save_image
                    save_image(adv_x.cpu(), str(trig_dir / f"{img_id}.png"))

                if (idx + 1) % 10 == 0:
                    print(f"[attack] {idx+1}/{args.num_images} done")
        else:
            for start in range(0, len(missing_indices), args.batch_size):
                batch_indices = missing_indices[start:start + args.batch_size]
                if not batch_indices:
                    break
                batch_paths = [sampled[i] for i in batch_indices]

                with torch.no_grad():
                    base_model.load_state_dict(base_state, strict=True)

                pil_imgs = [pil_load(p) for p in batch_paths]

                adv_list, info_list = pla_attack_batch(
                    model=base_model,
                    processor=base_processor,
                    pil_images=pil_imgs,
                    question_text=args.question,
                    target_text=args.target,
                    cfg=cfg,
                    record_losses=args.save_loss_curves,
                    device=args.device,
                    dtype=dtype,
                )

                for i, (adv_x, info, img_path) in enumerate(zip(adv_list, info_list, batch_paths)):
                    idx = batch_indices[i]
                    img_id = f"{idx:04d}"
                    triggers.append((img_id, adv_x.cpu(), info))

                    if args.save_loss_curves:
                        losses = info.pop("losses", None)
                        if losses is not None:
                            loss_path = loss_dir / f"{img_id}.json"
                            loss_path.write_text(json.dumps({"loss": losses}, ensure_ascii=False, indent=2), encoding="utf-8")

                    meta = {"image_id": img_id, "src_path": img_path, **info}
                    meta_by_id[img_id] = meta

                    if args.save_triggers:
                        from torchvision.utils import save_image
                        save_image(adv_x.cpu(), str(trig_dir / f"{img_id}.png"))

                done_count = min(batch_indices[-1] + 1, args.num_images)
                if done_count % 10 == 0:
                    print(f"[attack] {done_count}/{args.num_images} done")

        # cleanup base model to free VRAM before evaluation
        del base_model
        torch.cuda.empty_cache()

        if meta_by_id:
            trigger_meta = [meta_by_id[k] for k in sorted(meta_by_id.keys())]
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
