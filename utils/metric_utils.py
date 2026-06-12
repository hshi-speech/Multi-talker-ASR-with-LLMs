# Author: Hao at 2025-07-02
# metric_utils.py
# -----------------------------------------------------------
# Helper to load the Word Error Rate (WER) metric via `evaluate`
# and provide a `compute_metrics` callback for Hugging Face Trainer.
#
# Usage:
#   from metric_utils import build_compute_metrics
#   compute_metrics = build_compute_metrics(tokenizer, cache_dir)
#   trainer = Trainer(..., compute_metrics=compute_metrics)
# -----------------------------------------------------------

import evaluate
import numpy as np  # Optional, but HF Trainer sometimes passes numpy arrays
import concurrent.futures


def strip_prompt_prefix(pred_ids, bos_response_id: int, pad_id: int):
    """
    Mask everything up to and including <bos_response> in each prediction row.

    In instruct mode, generate() returns [bos] + prompt + response; references
    have the prompt masked out, so leaving the prompt words in the hypothesis
    inflates WER. Returns a new array (the input is not modified).
    """
    pred_ids = np.array(pred_ids, copy=True)
    for row in pred_ids:
        pos = np.nonzero(row == bos_response_id)[0]
        if pos.size:
            row[: pos[0] + 1] = pad_id
    return pred_ids


def compute_metrics(tokenizer, cache_dir: str = None, ignore_id: int = -100):
    """
    Return a `compute_metrics` function bound to the given tokenizer.

    Parameters
    ----------
    tokenizer:
        The tokenizer used for decoding model outputs.
    cache_dir: str, optional
        Where to cache the metric files.

    Returns
    -------
    callable
        A function `compute_metrics(pred)` compatible with HF Trainer.
    """
    metric = evaluate.load("wer", cache_dir=cache_dir)

    def computing_metrics(pred):
        """
        Compute WER between predictions and references.

        `pred` is the EvalPrediction object passed by HF Trainer.
        """
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 with pad_token_id to enable decoding (the eval loop may
        # pad concatenated prediction batches with -100 as well)
        pred_ids = np.array(pred_ids, copy=True)
        pred_ids[pred_ids == ignore_id] = tokenizer.pad_token_id
        label_ids[label_ids == ignore_id] = tokenizer.pad_token_id

        # Instruct mode: drop the generated prompt prefix (up to <bos_response>)
        # so hypotheses are compared on the response only, like the references.
        # In non-instruct tokenizers the token does not exist -> no-op.
        bosr_id = tokenizer.convert_tokens_to_ids("<bos_response>")
        if bosr_id is not None and bosr_id != getattr(tokenizer, "unk_token_id", None):
            pred_ids = strip_prompt_prefix(pred_ids, bosr_id, tokenizer.pad_token_id)

        # Decode
        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return computing_metrics

