# Geneformer DeepChem Integration & A/B Benchmarking Plan

## 1. Project Objective
To integrate the `Geneformer` transformer model into the DeepChem library as a `TorchModel` wrapper. The goal is to allow users to perform single-cell classification using DeepChem's `.fit()` and `.predict()` API while maintaining parity with the original Hugging Face implementation.

## 2. Core Constraints (The "DeepChem Way")
The implementation MUST follow the patterns established in `deepchem.models.torch_models.chembert`:
- **Architecture**: Inherit from `torch.nn.Module` for the internal network and `dc.models.TorchModel` for the wrapper.
- **Data Contract**: The model expects `dc.data.Dataset` objects. 
- **Internal Mapping**: The `_prepare_batch` method must handle:
    1. Converting Numpy arrays from DeepChem Shards into PyTorch Tensors.
    2. Generating `attention_mask` on the fly (padding to max sequence length in batch).
    3. Mapping the DeepChem `y` (labels) to the expected input for the Hugging Face `labels` argument.
- **Model Loading**: Must support `from_pretrained` logic using Hugging Face's `subfolder` argument to target specific Geneformer weights (e.g., `Geneformer-V1-10M`).

## 3. Implementation Plan
### Phase A: The Wrapper (To be placed in `deepchem/models/torch_models/`)
- **Backbone**: `BertForSequenceClassification` from the `transformers` library.
- **Freezing Logic**: Implement a mechanism to freeze the first $N$ layers (default=2) to match the official Geneformer paper's memory-saving strategy.

### Phase B: Data Parity
- **Source**: `ctheodoris/human_dcm_hcm_nf` (Hugging Face).
- **Subsampling**: 5,000 cells for Training, 1,000 for Testing (sampled with `seed=42`).
- **Standardization**: Both the Native HF run and the DeepChem Wrapper run MUST use the exact same subset of data.

### Phase C: A/B Testing (Benchmarking)
Compare two execution paths:
1. **Baseline**: Native Hugging Face `Trainer` + `BertForSequenceClassification`.
2. **Challenger**: DeepChem `Geneformer` wrapper + `dc.data.DiskDataset`.

**Metrics for Success**:
- Accuracy and Macro F1 score must be within $\pm 1\%$ between both implementations.
- Loss convergence curves should follow the same trajectory.

## 4. Hardware & Environment Context
- **GPU**: NVIDIA RTX 3090 (24GB VRAM).
- **Memory**: 100GB System RAM.
- **Storage**: 50GB Volume.
- **Optimizer**: AdamW with a learning rate of $5 \times 10^{-5}$ and Weight Decay of $0.01$.
- **Batch Size**: 12 (to match official implementation).

## 5. Troubleshooting for Copilot CLI
If an error occurs in `_prepare_batch` or during `.fit()`:
- Check if the tensor shapes are $(Batch, SequenceLength)$.
- Ensure `input_ids` are `LongTensor`.
- Verify that the `Geneformer` model is receiving `attention_mask`.

## 6. Validation Protocol (A/B Test Logic)
To ensure the DeepChem Wrapper is production-ready, we must verify the following:

### Step 1: Weight Initialization Parity
- Both models must load from `ctheodoris/Geneformer` (subfolder: `Geneformer-V1-10M`).
- We must verify that the `MISSING` keys (the classification head) are being initialized.

### Step 2: Input Formatting Logic
- **Native HF**: Must use a manual `data_collator` to pad sequences to 2048 and create `attention_masks`.
- **DeepChem**: Must use `dc.data.DiskDataset`. The wrapper's `_prepare_batch` must handle the padding and mask generation automatically. 
- *Success Criteria*: Copilot should verify that `input_ids` and `attention_mask` shapes are identical in both pipelines.

### Step 3: Optimization Parity
- Optimizer: `AdamW`
- Learning Rate: `5e-5`
- Weight Decay: `0.01`
- **Crucial**: The DeepChem wrapper must be checked to ensure it isn't applying double-dropout or extra normalization that the Native HF model isn't using.

### Step 4: Metric Comparison
- Run 1 Epoch on both.
- Compare `eval_accuracy` and `eval_macro_f1`.
- If DeepChem accuracy is significantly lower (>3% difference), check `_prepare_batch` for token ID shifts or masking errors.

## 7. Resource Registry (Local Paths)
- **Hugging Face Hub ID**: `ctheodoris/Geneformer`
- **Subfolder**: `Geneformer-V1-10M`
- **Local HF Train Data**: `./data/hf/train` (Format: HuggingFace Dataset)
- **Local HF Test Data**: `./data/hf/test`
- **Local DeepChem Train**: `./data/dc/train` (Format: DeepChem DiskDataset)
- **Local DeepChem Test**: `./data/dc/test`
- **Device**: `cuda` (RTX 3090, 24GB VRAM)

## 8. Model-Specific Architecture Notes
- **Input Scaling**: Geneformer does not use a standard tokenizer; it uses rank-normalized gene expression. The `input_ids` are already pre-computed Ensembl IDs.
- **Attention Masking**: The model is highly sensitive to the `attention_mask`. In `_prepare_batch`, ensure that `0` tokens (padding) are strictly masked out with `0` in the attention tensor.
- **Output Head**: We are using `BertForSequenceClassification`. The `loss` should be calculated internally by the Hugging Face model when `labels` are passed into the `forward()` call.