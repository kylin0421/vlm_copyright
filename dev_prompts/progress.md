# Progress

- Added `model_loader.py` with separate HF vs official LLaVA loading paths and a processor/model adapter for non-HF checkpoints.
- Updated PLA to perform model parameter updates (gradient ascent) during optimization; added `model_lr`, `model_grad_clip`, and `reg_lambda`.
- Updated CLI to expose new PLA knobs and loader selection, and removed duplicate base-model loading.
- Expanded model directory discovery to include non-HF LLaVA checkpoint layouts.
