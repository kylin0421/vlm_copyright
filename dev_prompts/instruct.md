# Reproduction tasks

- Check repo `vlm_copyright/` and read paper https://arxiv.org/abs/2502.16593.
- Fix PLA implementation: must update model parameters during optimization (current code only updates image).
- Fix model loading for `LLaVAs_todo`: HF `Auto...from_pretrained` fails. Use separate non-HF loading logic that matches official LLaVA v1.5 (non-HF) behavior, rather than converting checkpoints.
- Separate model loading logic into its own module so future ckpts can plug in their own loader.
- Clean up the codebase (small refactors, reduce duplication).
