import deepchem as dc
import numpy as np
from typing import Any, Dict, Optional, Tuple

# Because we restarted the runtime, this import will now work perfectly!
from deepchem.models.torch_models.hf_models import HuggingFaceModel

try:
    import torch
    from transformers import (
        BertConfig,
        BertForMaskedLM,
        BertForSequenceClassification
        # Notice we removed PreTrainedTokenizerFast completely!
    )
    has_torch = True
except ImportError:
    has_torch = False

# 1. Define the class directly in the notebook memory
class Geneformer(HuggingFaceModel):
    def __init__(self,
                 task: str,
                 model_path: str = 'ctheodoris/Geneformer',
                 subfolder: str = 'Geneformer-V1-10M',
                 n_tasks: int = 1,
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs):

        if not has_torch: raise ImportError("Geneformer requires PyTorch and Transformers.")
        self.n_tasks = n_tasks

        # ---------------------------------------------------------
        # THE FIX: Bypass the text tokenizer completely and grab
        # the padding ID directly from the Hugging Face config.
        # ---------------------------------------------------------
        hf_config = BertConfig.from_pretrained(model_path, subfolder=subfolder, **(config or {}))
        self.pad_token_id = getattr(hf_config, 'pad_token_id', 0)

        if task == 'mlm':
            model = BertForMaskedLM.from_pretrained(model_path, subfolder=subfolder, config=hf_config)
        elif task == 'classification':
            if n_tasks == 1:
                hf_config.problem_type = 'single_label_classification'
            else:
                hf_config.problem_type = 'multi_label_classification'
                hf_config.num_labels = n_tasks

            model = BertForSequenceClassification.from_pretrained(
                model_path,
                subfolder=subfolder,
                config=hf_config,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"Invalid task '{task}'.")

        # Pass tokenizer=None since we handle pre-featurized arrays
        super(Geneformer, self).__init__(model=model, task=task, tokenizer=None, config=hf_config.to_dict(), **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, ...]):
        """Prepare a Geneformer batch with fixed sequence length (2048)."""
        if len(batch) == 4:
            X, y, w, _ = batch
        elif len(batch) == 3:
            X, y, w = batch
        else:
            raise ValueError("Expected batch to have 3 or 4 items: (X, y, w[, ids]).")

        input_ids = X[0] if isinstance(X, (list, tuple)) else X
        input_ids_t = torch.as_tensor(input_ids, device=self.device).long()

        if input_ids_t.dim() == 1:
            input_ids_t = input_ids_t.unsqueeze(0)

        seq_len = input_ids_t.shape[1]
        if seq_len > 2048:
            input_ids_t = input_ids_t[:, :2048]
        elif seq_len < 2048:
            pad_width = 2048 - seq_len
            input_ids_t = torch.nn.functional.pad(input_ids_t, (0, pad_width), value=0)

        attention_mask_t = (input_ids_t != 0).long()
        inputs_dict = {
            'input_ids': input_ids_t,
            'attention_mask': attention_mask_t,
        }

        labels = None
        if y is not None:
            y_arr = y[0] if isinstance(y, (list, tuple)) else y
            labels = torch.as_tensor(y_arr, device=self.device)
            labels = labels.long() if self.n_tasks == 1 else labels.float()
            if self.n_tasks == 1 and labels.dim() > 1:
                labels = labels.reshape(-1)

        weights = None
        if w is not None:
            weights = torch.as_tensor(w, dtype=torch.float, device=self.device)

        return inputs_dict, labels, weights

# 2. Immediately test the class we just defined
print("Testing wrapper instantiation...")
try:
    test_model = Geneformer(task='classification', n_tasks=3)
    print("✅ Geneformer wrapper successfully loaded and weights downloaded in memory!")
except Exception as e:
    print(f"❌ Initialization failed: {e}")