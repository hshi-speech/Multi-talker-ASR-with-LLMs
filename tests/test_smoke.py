# Created by Hao at 2026-06-12 (added during code review, see REVIEW.md)
"""Smoke tests: label-serialization round-trip, collator alignment, tiny forward pass.

Run with:  pytest tests/test_smoke.py
No GPU, no network, no pretrained checkpoints required.
"""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from utils.split_labels_by_sc import split_k_speakers_and_lengths  # noqa: E402
from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding  # noqa: E402

PAD, START, EOS, SC = 0, 1, 2, 5
IGNORE = -100


# ---------------------------------------------------------------------------
# Minimal stand-ins for the tokenizer / feature-extractor / processor
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    pad_token_id = PAD

    def pad(self, features, return_tensors=None):
        seqs = [f["input_ids"] for f in features]
        max_len = max((len(s) for s in seqs), default=0)
        ids = torch.full((len(seqs), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1

        class _Batch(dict):
            @property
            def attention_mask(self):
                return self["attention_mask"]

        return _Batch(input_ids=ids, attention_mask=mask)


class _FakeFeatureExtractor:
    model_input_names = ["input_values"]

    def pad(self, features, return_tensors=None):
        seqs = [f["input_values"] for f in features]
        max_len = max(len(s) for s in seqs)
        x = torch.zeros((len(seqs), max_len))
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return {"input_values": x, "attention_mask": mask}


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    feature_extractor = _FakeFeatureExtractor()
    model_input_names = ["input_values"]


class _FakeConfig:
    ignore_token_id = IGNORE
    model_type = "wavlm"


def _make_collator():
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=_FakeProcessor(),
        decoder_start_token_id=START,
        decoder_end_token_id=EOS,
        config=_FakeConfig(),
    )


# ---------------------------------------------------------------------------
# 1. Label-serialization round-trip
# ---------------------------------------------------------------------------
def test_split_k_speakers_round_trip():
    """Serialized [spk1 <sc> spk2] labels split back into the original segments."""
    spk1 = [[10, 11, 12], [30, 31]]
    spk2 = [[20, 21], [40, 41, 42, 43]]
    rows = []
    for a, b in zip(spk1, spk2):
        rows.append(a + [SC] + b)
    max_len = max(len(r) for r in rows)
    labels = torch.full((len(rows), max_len), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        labels[i, : len(r)] = torch.tensor(r)

    segs, lens = split_k_speakers_and_lengths(
        labels=labels,
        k_speakers=2,
        sep_id=SC,
        pad_token_id=PAD,
        end_token_id=PAD,
        ignore_id=IGNORE,
        allow_empty_segment=False,
    )
    assert len(segs) == len(lens) == 2
    for spk_idx, ref in enumerate([spk1, spk2]):
        for b, ref_row in enumerate(ref):
            n = lens[spk_idx][b].item()
            assert n == len(ref_row)
            assert segs[spk_idx][b, :n].tolist() == ref_row


def test_split_k_speakers_wrong_separator_count_raises():
    labels = torch.tensor([[10, 11, 12]])  # no <sc> but k_speakers=2
    with pytest.raises(ValueError):
        split_k_speakers_and_lengths(
            labels=labels, k_speakers=2, sep_id=SC, pad_token_id=PAD,
            end_token_id=PAD, ignore_id=IGNORE, allow_empty_segment=False,
        )


# ---------------------------------------------------------------------------
# 2. Collator alignment on tiny dummy data
# ---------------------------------------------------------------------------
def test_collator_teacher_forcing_alignment():
    coll = _make_collator()
    feats = [
        {"input_values": [0.1] * 20, "labels": [START, 10, 11, SC, 13],
         "prompt_ids": [], "prompt_token_len": 0},
        {"input_values": [0.1] * 10, "labels": [START, 20, SC, 21],
         "prompt_ids": [], "prompt_token_len": 0},
    ]
    batch = coll(feats)
    di, la = batch["decoder_input_ids"], batch["labels"]
    assert di.shape == la.shape

    # longest row: full teacher forcing incl. last token, EOS after it
    assert di[0].tolist() == [START, 10, 11, SC, 13]
    assert la[0].tolist() == [10, 11, SC, 13, EOS]
    # shorter row: EOS right after the transcript, rest ignored
    assert di[1].tolist() == [START, 20, SC, 21, PAD]
    assert la[1].tolist() == [20, SC, 21, EOS, IGNORE]

    # invariant: input at step t is the label of step t-1 (when both supervised)
    for r in range(la.shape[0]):
        for t in range(1, la.shape[1]):
            if la[r, t] != IGNORE and la[r, t - 1] != IGNORE:
                assert di[r, t].item() == la[r, t - 1].item()


def test_collator_prompt_masking():
    coll = _make_collator()
    # rows share prompt length 2 (the batch-uniform assumption, see ISSUE-07)
    feats = [
        {"input_values": [0.1] * 20, "labels": [START, 7, 8, 10, SC, 11],
         "prompt_ids": [7, 8], "prompt_token_len": 2},
        {"input_values": [0.1] * 10, "labels": [START, 7, 8, 20, SC, 21],
         "prompt_ids": [7, 8], "prompt_token_len": 2},
    ]
    batch = coll(feats)
    la = batch["labels"]
    # prompt positions excluded from the CE loss
    assert (la[:, :2] == IGNORE).all()
    # response tokens still supervised
    assert la[0, 2:].tolist() == [10, SC, 11, EOS]
    assert batch["prompt_ids"].tolist() == [[7, 8], [7, 8]]


# ---------------------------------------------------------------------------
# 2b. Eval-WER prompt stripping (ISSUE-17)
# ---------------------------------------------------------------------------
def test_strip_prompt_prefix():
    import numpy as np
    from utils.metric_utils import strip_prompt_prefix

    BOSR = 9
    preds = np.array([
        [1, 7, 8, BOSR, 10, 11, 12],   # prompt 7,8 then response
        [1, 20, 21, 22, 23, 24, 25],   # no <bos_response> -> untouched
    ])
    out = strip_prompt_prefix(preds, bos_response_id=BOSR, pad_id=PAD)
    assert out[0].tolist() == [PAD, PAD, PAD, PAD, 10, 11, 12]
    assert out[1].tolist() == preds[1].tolist()
    # input must not be modified in place
    assert preds[0, 0] == 1


# ---------------------------------------------------------------------------
# 3. Forward pass (hybrid CE+CTC loss) on a tiny randomly-initialized model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model():
    from transformers import WavLMConfig, LlamaConfig
    from transformers.models.speech_encoder_decoder.configuration_speech_encoder_decoder import (
        SpeechEncoderDecoderConfig,
    )
    from modeling_speech_encoder_decoder_llama import SpeechEncoderDecoderModelLlama

    enc_cfg = WavLMConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, conv_dim=(32, 32), conv_stride=(5, 2),
        conv_kernel=(10, 3), num_feat_extract_layers=2,
        add_adapter=True, num_adapter_layers=2, adapter_stride=2,
        output_hidden_size=32, mask_time_prob=0.0, mask_feature_prob=0.0,
        layerdrop=0.0, num_buckets=16, max_bucket_distance=40,
    )
    dec_cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=PAD, bos_token_id=START, eos_token_id=EOS,
        is_decoder=True, attention_dropout=0.0,
    )
    cfg = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(enc_cfg, dec_cfg)
    cfg.decoder_start_token_id = START
    cfg.pad_token_id = PAD
    cfg.eos_token_id = EOS
    cfg.ignore_token_id = IGNORE
    cfg.sc_token_id = SC
    cfg.instruct = False
    cfg.decoder.instruct = False
    cfg.talker_ctc = True
    cfg.talker_ctc_refine = False
    cfg.talker_numbers = 2
    cfg.separator_hidden = 16
    cfg.train_mode = "hybrid"
    cfg.ctc_alpha = 0.7
    cfg.ctc_bridge = False
    cfg.decoder_cross_attention = False

    torch.manual_seed(0)
    model = SpeechEncoderDecoderModelLlama(config=cfg)
    # mirror finetune_asr.py / inference_asr.py step 7: the vendored generation code
    # reads generation_config.forced_decoder_ids, which newer GenerationConfig
    # versions no longer define — the entry points set it to None explicitly.
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.eval()
    return model


def test_tiny_forward_hybrid_loss(tiny_model):
    coll = _make_collator()
    feats = [
        {"input_values": [0.01] * 4000, "labels": [START, 10, 11, SC, 13],
         "prompt_ids": [], "prompt_token_len": 0},
        {"input_values": [0.01] * 3000, "labels": [START, 20, SC, 21],
         "prompt_ids": [], "prompt_token_len": 0},
    ]
    batch = coll(feats)
    with torch.no_grad():
        out = tiny_model(
            inputs=batch["input_values"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
            return_dict=True,
        )
    assert out.loss is not None
    assert torch.isfinite(out.loss), f"loss is not finite: {out.loss}"
    # logits cover speech frames + text positions, over the decoder vocab
    assert out.logits.dim() == 3
    assert out.logits.shape[0] == 2
    assert out.logits.shape[-1] == tiny_model.decoder.config.vocab_size
    assert out.logits.shape[1] > batch["labels"].shape[1]  # speech prefix inserted


def test_tiny_generate_greedy(tiny_model):
    with torch.no_grad():
        out = tiny_model.generate(
            inputs=torch.zeros(1, 4000),
            prompt_ids=None,
            max_length=8,
            num_beams=1,
            synced_gpus=False,
            use_cache=True,
        )
    assert out.dim() == 2 and out.shape[0] == 1
    assert out.shape[1] <= 8
    assert out[0, 0].item() == START
