# Geneformer Integration Progress Log

## Last Updated
- Date: 2026-03-16
- Updated via: Copilot CLI session

## What Was Completed
- Implemented `Geneformer._prepare_batch()` in `deepchem/models/torch_models/geneformer.py`.
- Enforced `input_ids` tensor shape to `(batch, 2048)`.
- Added truncation when sequence length is `> 2048`.
- Added right-padding with `0` when sequence length is `< 2048`.
- Ensured `input_ids` are moved to `self.device` and cast to `.long()`.
- Implemented `attention_mask` creation from `input_ids` with rule:
	- non-zero token -> `1`
	- zero token (padding) -> `0`
- Returned DeepChem-compatible tuple format:
	- `(inputs_dict, labels, weights)`
	- where `inputs_dict = {"input_ids": ..., "attention_mask": ...}`
- Kept label handling aligned with DeepChem classification conventions:
	- single-task classification labels as `long`
	- multi-task labels as `float`
- Kept `weights` conversion to float tensor on `self.device`.

## `scripts/bench_native.py` Work Completed (Copilot CLI)
- Created `scripts/bench_native.py` as the native Hugging Face baseline benchmark for Geneformer.
- Added dataset loading from disk using:
- `--train-path` (default `./data/hf/train`)
- `--test-path` (default `./data/hf/test`)
- Added dataset schema checks:
- requires `input_ids`
- accepts `label` or `labels`
- Added row normalization to standard format:
- `input_ids` coerced to `int64`
- `attention_mask` created from non-zero `input_ids` if missing
- labels normalized into `labels`
- Added class-count inference from normalized train labels.
- Added model setup for native baseline:
- loads `BertForSequenceClassification` from `ctheodoris/Geneformer` with subfolder `Geneformer-V1-10M`
- uses `ignore_mismatched_sizes=True` for classifier head init
- Added freezing logic for first 2 encoder layers.
- Added training config for parity target:
- `fp16=True`
- `learning_rate=5e-5`
- `per_device_train_batch_size=12`
- `per_device_eval_batch_size=12`
- `num_train_epochs=1`
- `save_strategy="no"`, `logging_strategy="no"`, `report_to=[]`
- Added metric computation and printout:
- Accuracy (`eval_accuracy`)
- Macro F1 (`eval_macro_f1`)
- Added CLI output directory support via `--output-dir` (default `./runs/bench_native`).

## `scripts/bench_native.py` Validation Performed
- `python -m py_compile scripts/bench_native.py` passed.
- `python scripts/bench_native.py --help` works.
- Running without local HF datasets raises explicit:
- `FileNotFoundError: Expected datasets at 'data/hf/train' and 'data/hf/test'.`
- Added runtime guard for CUDA:
- raises if `torch.cuda.is_available()` is false because `fp16=True` is enabled.

## Current `_prepare_batch` Contract
- Accepts either batch length `3` or `4`:
	- `(X, y, w)`
	- `(X, y, w, ids)`
- Raises a clear `ValueError` for any other batch format.

## Validation Performed
- File-level diagnostics were run for:
	- `deepchem/models/torch_models/geneformer.py`
- Result:
	- No syntax/type errors reported by editor diagnostics.

## Important Notes
- The current file still includes notebook-style test code at module scope:
	- `print("Testing wrapper instantiation...")`
	- immediate model initialization in a `try/except`
- This should be removed before upstream integration/PR because import-time side effects are not DeepChem style.

## Remaining Work (Next Session Start Here)
- Refactor `geneformer.py` to DeepChem production style:
	- remove notebook comments and import-time test execution
	- keep only class/module definitions
- Add unit tests for `_prepare_batch` covering:
	- exact length 2048
	- shorter than 2048 (padding)
	- longer than 2048 (truncation)
	- attention mask correctness
	- single-task and multi-task label dtype behavior
- Verify integration with `dc.data.DiskDataset` pipeline (real mini-batch pass through `.fit()`).
- Implement planned layer-freezing option (freeze first N transformer layers, default 2).
- Run A/B parity checks versus Hugging Face baseline per `geneformer_plan.md`.
- Run `scripts/bench_native.py` end-to-end on prepared local HF data and capture baseline metrics.
- Add/prepare challenger benchmark script (`bench_dc.py`) using DeepChem `DiskDataset` with matching hyperparameters.
- Record baseline vs challenger metric delta and confirm parity target (+/-1% accuracy and macro F1).

## Quick Resume Checklist
- Open `deepchem/models/torch_models/geneformer.py`.
- Remove notebook-only code and clean imports.
- Create/extend tests under DeepChem torch model test suite.
- Run targeted tests and a short training smoke test.
- Verify datasets exist at `./data/hf/train` and `./data/hf/test`.
- Run `python scripts/bench_native.py` and log printed Accuracy/Macro F1.
- Start DeepChem-side benchmark script with the same subset and optimizer settings.

