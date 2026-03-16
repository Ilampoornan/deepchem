# Geneformer Progress

## What we completed
- Created `scripts/bench_native.py` for native Hugging Face Geneformer fine-tuning.
- Script now:
  - Loads `./data/hf/train` and `./data/hf/test` via `datasets.load_from_disk`
  - Initializes `BertForSequenceClassification` from `ctheodoris/Geneformer` with subfolder `Geneformer-V1-10M`
  - Freezes the first 2 BERT encoder layers
  - Uses `TrainingArguments` with:
    - `fp16=True`
    - `learning_rate=5e-5`
    - `per_device_train_batch_size=12`
    - `num_train_epochs=1`
  - Trains and evaluates with Accuracy + Macro F1
  - Prints:
    - `Accuracy: ...`
    - `Macro F1: ...`

## Validation done
- `python -m py_compile scripts/bench_native.py` passed.
- `python scripts/bench_native.py --help` works.
- Running without local data gives explicit:
  `FileNotFoundError: Expected datasets at 'data/hf/train' and 'data/hf/test'.`

## Very next step
1. Ensure local datasets exist at:
   - `./data/hf/train`
   - `./data/hf/test`
2. Run:
   `python scripts/bench_native.py`
3. Capture and record the printed `Accuracy` and `Macro F1`.
