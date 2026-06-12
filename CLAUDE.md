# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LLM-based **multi-talker ASR** with **serialized output training (SOT)** on LibriMix mixtures.
A WavLM front-end encodes an *N*-speaker mixture; an optional LSTM separator splits encoder
features into *N* per-speaker streams supervised by per-speaker CTC heads; a LLaMA decoder
generates all *N* transcripts as one serialized token sequence joined by a `<sc>` token. The
training objective is a hybrid of decoder cross-entropy and per-speaker CTC.

Companion docs (open in a browser): `WORKFLOW.html` (architecture, data flow, full CLI flag
reference) and `CODE_GUIDE.html`. `README.md` documents every stage and flag in detail.

## Commands

Everything runs through `run.sh`, which is stage-based (`stage`/`stop_stage`, 1–6). There are
**no tests, no linter, and no build step** — this is a research training pipeline driven by SLURM.

```bash
bash run.sh stage=1 stop_stage=1 ...   # build HF DatasetDict from LibriMix wav.scp/text → datasets/
bash run.sh stage=2 stop_stage=2 ...   # pair wavlm-large + LLaMA, add special tokens → dump/
bash run.sh stage=3 stop_stage=3 ...   # train (finetune_asr.py, torch.distributed.launch over all GPUs)
bash run.sh stage=4 stop_stage=4 ...   # attention-decoder inference + WER (inference_asr.py, 1 GPU)
bash run.sh stage=5 stop_stage=5 ...   # CTC-only decoding (--ctc_decoding=true)
bash run.sh stage=6 stop_stage=6 ...   # multi-GPU sharded inference / pseudo-labels (inference_asr_gpus.py)
```

`run.sh` requires a long list of flags (no defaults for most) and **constructs `output_dir`
deterministically from the flag values** (e.g. `mode_hybrid-wavlm-Llama-3.2-1B-encoder_freeze-...-libri2mix_clean`).
Stages 4/5/6 must be invoked with the *same* flags that produced the training `output_dir`, or
they will look in the wrong directory. See `README.md` for full per-stage flag lists.

The launcher scripts live in `scripts/` (renamed from `slurm/`; **no SLURM anymore**): each
`*submit*.sh` runs its configurations sequentially on the local machine by calling `../run.sh`
directly — defaults for all unlisted flags now live in `run.sh` itself, so `run.sh` can also be
invoked standalone with only the flags that differ. Edit the flags at the top of e.g.
`scripts/sot_submit_3b.sh` and run it with `bash`; limit GPUs with `CUDA_VISIBLE_DEVICES`.
Variants: `sot_*` (SOT only), `ctc_*` (CTC only), `crossatt_*` (cross-attention adapters),
`cross_gate_*` (gated/LoRA cross-attention). `run.sh` falls back to the `python3` on PATH when
`virtual_env` is empty/unset, and `cache_dir` defaults to `~/.hf_cache`.
(`scripts/template.slurm` is a retired reference file from the old cluster setup.)

> `requirements.txt` is an incomplete `pip freeze` and **omits the core ML deps**. You also need
> at least: `torch transformers peft accelerate datasets safetensors librosa soundfile sentencepiece jiwer`.

## Architecture

### The glue model (`models/modeling_speech_encoder_decoder_llama.py`)
`SpeechEncoderDecoderModelLlama` is a **fork of HuggingFace's `SpeechEncoderDecoderModel`**, not a
subclass — it is heavily customized and the upstream version will not behave the same way. Key facts:

- `encoder_outputs[0]` is the **downsampled** adapter output (fed to the decoder, projected via
  `enc_to_dec_proj`); `encoder_outputs[1]` is the **pre-adapter** WavLM hidden states, used to feed
  the separator and CTC heads. Confusing these two is the most common source of shape bugs.
- The separator + `serialized_ctc` ModuleList (one `CTC` head per speaker) only exist when
  `talker_ctc=True`. CTC blank id = `decoder.vocab_size + 1`.
- Loss lives in a single `HybridLoss` (`models/losses.py`); `train_mode` (`attention`/`ctc`/`hybrid`)
  selects which terms are active. `ctc_alpha` (default 0.7) weights CE vs CTC.
- Speech embeddings are inserted *inside the LLaMA decoder forward*; labels are left-padded with
  `ignore_token_id` (-100) over the `speech_len` positions to realign with logits. Prompt-prefix
  masking and EOS insertion happen in the **data collator** (per-sample), not in the model.
- Generation is delegated to mix-ins, not HF's default: `GenerationMixin_Instruct`
  (`utils/generation_utils.py`) for attention decoding and `GenerationMixin_CTC`
  (`utils/generation_ctc_utils.py`, via `generate_ctc`/`forward_ctc`) for CTC decoding.

### `models/` import convention — important
Modules in `models/` import each other by **bare module name** (e.g.
`from modeling_llama import LlamaForCausalLM`, `from separator import Separator`), *not* by package
path (`models.modeling_llama`). This works only because entry points and loaders insert the
`models/` directory onto `sys.path` before importing (see the `sys.path.insert` blocks in
`src/model_loader.py`, `utils/create_from_pretrained.py`, and the glue model itself). When adding a
new model file, follow this same flat-import style and ensure any new entry point inserts `models/`
onto `sys.path`.

### Cross-attention adapters (optional)
When `decoder_cross_attention=True`, a per-layer adapter ModuleList is injected into the decoder.
Variant is chosen by `decoder_cross_attention_type`: `tiny`, `gatetiny`, `adapgatetiny` (adaptive-rank
LoRA, tuned via `r_max`/`lora_alpha`), `ctcaware`. Memory source is `decoder_cross_attention_feature`:
`mix` (raw WavLM) or `sep` (per-speaker separator features). Each variant is its own
`*_crossatt_module.py` file.

### Two independent LoRA groups — do not conflate
1. **Self-attention LoRA** — PEFT-injected into decoder `k/q/v/o_proj` by `src/insert_adapter_decoder.py`
   when `adapter_only_decoder=True`. Controlled by `selfattn_lora_r`/`selfattn_lora_alpha`/`selfattn_lora_dropout`.
   **Merged back into the base weights after training** by `utils/merge_adapter.py` (run.sh stage 3
   saves `model_unmerge.safetensors` first, then merges).
2. **Cross-attention LoRA** — inside the `adapgatetiny` adapter. Controlled by `r_max`/`lora_alpha`.

### `train_mode` decides freezing and saving
- `ctc`: no adapters inserted; whole model frozen then selected params re-enabled; saves `model.safetensors`.
- `attention`/`hybrid`: model frozen, LoRA inserted (if `adapter_only_decoder`), then params re-enabled.
- Freezing is layered: `freeze_model` freezes everything, then `unfreeze_selected_params` re-enables
  based on `train_mode` plus `partial_{encoder,decoder,others}_unfreeze` (comma-separated name substrings,
  e.g. `partial_others_unfreeze="enc_to_dec_proj"`).

### `finetune_asr.py` orchestration (stage 3 entry)
Numbered steps drive the run: load args/dataset → build `config` and copy the `talker_*`/`ctc_*`/
`decoder_cross_attention*` model flags onto it → load feature extractor, tokenizer, AED model →
optionally load a pretrained separator (`pretrain_separator_path`) → freeze + insert adapters +
unfreeze → vectorize dataset (`utils/vectorized_dataset_utils.py`) → build processor → `DataCollatorSpeechSeq2SeqWithPadding`
→ subclassed `Seq2SeqTrainer` (`src/trainer_seq2seq.py`).

### Directory map
- `src/` — training infra: `arguments.py` (all `ModelArguments`/`DataTrainingArguments`), loaders
  (config/tokenizer/feature_extractor/model/dataset), `data_collator.py`, `trainer_seq2seq.py`,
  `insert_adapter_decoder.py`.
- `models/` — model code (see import convention above).
- `utils/` — dataset prep (`generate_dataset.py`), model creation (`create_from_pretrained.py`),
  freeze/unfreeze, generation mix-ins, WER (`compute-wer.py`, `metric_utils.py`), adapter merge,
  safetensors fixups.
- `scripts/` — local launcher scripts (former SLURM submitters). Generated outputs (gitignored): `datasets/`, `dump/`, `exp*/`.

## Conventions
- Files carry a `# Created by Hao at <date>` header; keep that style for new files.
- Special tokens added at stage 2: `<sc>` always; the full Instruct set (`<bos_prompt>`,
  `<eos_prompt>`, `<bos_speech>`, `<eos_speech>`, `<bos_response>`, `<eos_response>`) only when
  `instruct=true`.
