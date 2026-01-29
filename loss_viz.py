from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_losses(folder: Path) -> List[List[float]]:
    losses = []
    for p in sorted(folder.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = data.get("loss", None)
        if isinstance(seq, list) and seq:
            losses.append([float(x) for x in seq])
    return losses


def mean_curve(losses: List[List[float]]) -> List[float]:
    if not losses:
        return []
    min_len = min(len(x) for x in losses)
    if min_len == 0:
        return []
    out = []
    for i in range(min_len):
        out.append(sum(x[i] for x in losses) / len(losses))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_dir", type=str, required=True, help="Folder with per-image loss JSON files")
    ap.add_argument("--out_png", type=str, required=True, help="Output PNG path")
    args = ap.parse_args()

    loss_dir = Path(args.loss_dir)
    out_png = Path(args.out_png)

    losses = load_losses(loss_dir)
    if not losses:
        raise SystemExit(f"No valid loss JSON files found under {loss_dir}")

    avg = mean_curve(losses)
    if not avg:
        raise SystemExit("Loss curves are empty after alignment.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(avg, linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Average loss")
    plt.title(f"Mean loss over {len(losses)} samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
