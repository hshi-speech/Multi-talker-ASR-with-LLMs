# Code Review — Multi-talker ASR with LLMs

Reviewed: 2026-06-12. Scope: training (`finetune_asr.py`, `src/`, `models/`,
`utils/`), inference (`inference_asr.py`, `inference_asr_gpus.py`), data pipeline
(`utils/generate_dataset.py`, `utils/vectorized_dataset_utils.py`,
`src/data_collator.py`), and the driver scripts (`run.sh`, `slurm/`).

## Overview

**Pipeline.** `run.sh` is stage-based (1–6):

1. `utils/generate_dataset.py` builds a HF `DatasetDict` (columns `id`, `audio`
   (16 kHz), `text` (serialized multi-speaker transcript containing `<sc>`),
   `prompt` (one fixed instruction string)) from LibriMix `wav.scp`/`text`.
2. `utils/create_from_pretrained.py` pairs `microsoft/wavlm-large` with a LLaMA
   decoder into `SpeechEncoderDecoderModelLlama`, adds special tokens
   (`<sc>`, `<pad>`, and the Instruct set when `--instruct`), resizes the decoder
   embeddings, and writes model + tokenizer + feature extractor to `dump/`.
3. `finetune_asr.py` trains. Flow: load args → load DatasetDict → load config and
   copy the `talker_*`/`ctc_*`/`decoder_cross_attention*`/`r_max`/`lora_alpha`
   model flags onto it → load feature extractor / tokenizer / AED model →
   optional partial separator+CTC load → freeze, optionally PEFT-inject
   self-attention LoRA (`src/insert_adapter_decoder.py`), selectively unfreeze
   (`utils/unfreeze_utils.py`) → vectorize dataset
   (`utils/vectorized_dataset_utils.py`: feature extraction + prompt/label
   tokenization + `prompt_token_len`) → `DataCollatorSpeechSeq2SeqWithPadding`
   (pads, builds `decoder_input_ids`, inserts EOS, masks the prompt prefix with
   −100) → custom `Seq2SeqTrainer` (`src/trainer_seq2seq.py`, vendored from HF
   ~v4.47 with a PCGrad block in `training_step`). After training, self-attn
   LoRA is merged by `utils/merge_adapter.py` into `model.safetensors`.
4. `inference_asr.py` re-loads everything from the training `output_dir`,
   re-vectorizes the eval split and decodes sample-by-sample via the custom
   generation mix-in (`utils/generation_utils.py`); WER via
   `utils/compute-wer.py`.
5. Same as 4 but `--ctc_decoding=true` → `generate_ctc`
   (`utils/generation_ctc_utils.py` → `forward_ctc`, greedy CTC per speaker).
6. `inference_asr_gpus.py` shards the dataset across ranks for multi-GPU
   decoding / pseudo-labeling.

**Model.** `models/modeling_speech_encoder_decoder_llama.py` (forked HF
`SpeechEncoderDecoderModel`): `encoder_outputs[0]` = adapter-downsampled WavLM
output → `enc_to_dec_proj` → inserted as embeddings *inside* the LLaMA decoder
forward (after `<bos_speech>` in instruct mode, after BOS otherwise);
`encoder_outputs[1]` = pre-adapter WavLM states → LSTM `Separator` → N
per-speaker streams → per-speaker `CTC` heads (blank id = `decoder.vocab_size`,
head dim = `vocab_size+1`). Loss = `HybridLoss` (`models/losses.py`):
`alpha·CE + (1−alpha)·mean(per-speaker CTC)` with fixed-order (head i ↔
speaker i) assignment; labels are left-padded with −100 over the `speech_len`
positions so CE logits/labels align without an extra shift. Optional per-layer
cross-attention adapters (`tiny`/`gatetiny`/`adapgatetiny`/`ctcaware`) and CTC
bridge variants.

**Label serialization.** Speakers are serialized FIFO in the dataset `text`
joined by `<sc>` (no PIT — the old PIT scaffolding was removed). Training CE
sees the full serialized sequence; per-speaker CTC labels are obtained by
splitting `decoder_input_ids` at `<sc>` (`utils/split_labels_by_sc.py`).
Inference decodes the same serialized format and splits on `<sc>` for scoring —
consistent with training as long as the dataset order is itself consistent.

## Issues found

Severity: **Critical** = wrong results/crashes, **Major** = silent incorrectness
or train/inference mismatch, **Minor** = style/dead code/hard-coded paths.

---

**ISSUE-01 — `prompt_token_len` is stripped by the Trainer before the collator → training crashes on the first batch.**
`src/data_collator.py:94-101` + `src/trainer_seq2seq.py:890-938` + `finetune_asr.py:157-169`. **Critical.**
The collator requires `feature["prompt_token_len"]` (added in
`utils/vectorized_dataset_utils.py:105`), but `remove_unused_columns` defaults
to `True` and `_remove_unused_columns()` keeps only columns matching
`model.forward`'s signature (`input_values`, `labels`, `prompt_ids`, …) —
`prompt_token_len` is not in the signature, so it is dropped from the dataset
and the collator raises `KeyError: 'prompt_token_len'` on the first train step.
(This was introduced when prompt masking was moved from the model into the
collator; commit `936496c "fix bug by cluade without checking"`.)
*Fix:* set `training_args.remove_unused_columns = False` in `finetune_asr.py`
(the collator only reads the keys it needs, so extra columns are harmless).

**ISSUE-02 — hard-coded `output_dir` override hijacks stages 5–6.**
`run.sh:319` (and `run_librispeechmix.sh:215`). **Critical.**
`output_dir=/lustre/users/shi/.../ckpt_cheking` is unconditionally re-assigned
between stage 4 and stage 5, so CTC decoding (stage 5) and multi-GPU inference
(stage 6) load the model from, and write results to, a debugging path on
another user's cluster instead of the deterministically-constructed training
`output_dir`. *Fix:* delete the override lines.

**ISSUE-03 — multi-GPU inference puts the model on `cuda:0` for every rank.**
`inference_asr_gpus.py:136-138` vs `:184`. **Critical.**
`model.to("cuda")` runs *before* `setup_distributed()` calls
`torch.cuda.set_device(local_rank)`, so all ranks place the model on `cuda:0`
while input tensors go to `cuda:local_rank` → device-mismatch crash (or, at
best, all ranks contending on GPU 0). *Fix:* initialize distributed first and
move the model to the per-rank device.

**ISSUE-04 — vendored Trainer is missing imports; `NameError` on auto-resume and other paths.**
`src/trainer_seq2seq.py:836,852,860,1173,1240,1284,1325-1326,1359-1361,1391,1668,1794` (+ `:520`). **Critical.**
The file was copied from HF Trainer but several imports were dropped:
`TRAINER_STATE_NAME`, `hf_hub_utils`, `unwrap_model`, `deepspeed_init`,
`deepspeed_load_checkpoint`, `_is_peft_model`, `hp_params`,
`tpu_spmd_dataloader`, and `torch.distributed as dist`. The
`TRAINER_STATE_NAME` ones are on the **checkpoint-resume path**, which
`utils/checkpoint_checking_utils.resume_or_raise` triggers automatically
whenever a checkpoint exists in `output_dir` — resuming a crashed run dies
with `NameError`. The others break `push_to_hub`, FSDP, DeepSpeed resume, HP
search, and `load_best_model_at_end` under DDP. *Fix:* add the missing imports
(all names verified importable from transformers 4.49 / huggingface_hub).

**ISSUE-05 — undeclared/incompatible dependency versions; entry points fail at import.**
`requirements.txt` (whole file), `utils/generation_utils.py:30-70`, `utils/generation_ctc_utils.py:30-70`. **Critical.**
`requirements.txt` is an incomplete freeze: it pins neither `torch` nor
`transformers` and omits `peft`, `evaluate`, and `typeguard` (all imported by
the code). The supported transformers window turned out to be narrow in *both*
directions: with ≥4.55 the vendored generation/trainer files fail at import
(`QuantizedCacheConfig`, `_crop_past_key_values` removed — with 4.57 installed
**every entry point fails at import**); with ≤4.49 everything imports but
`PreTrainedModel` still inherits HF's `GenerationMixin`, which **shadows the
fork's `GenerationMixin_Instruct` in the MRO** — HF's generic
`prepare_inputs_for_generation` then silently drops the encoder
`attention_mask` and `generate()` crashes (verified empirically). *Fix:* pin
`transformers==4.53.3` (verified: imports + forward + generate all pass) and
declare the missing core deps in `requirements.txt`.

**ISSUE-06 — off-by-one in the collator: the longest sample's last token never enters `decoder_input_ids`; EOS is predicted from a pad embedding.**
`src/data_collator.py:70-92`. **Major.**
`decoder_input_ids` is built as `[start, labels[:, :-1]]` and then a pad column
is appended, while EOS is inserted into `labels` at the first padding position.
For the longest row(s) in each batch the first padding position is the appended
column, so teacher forcing becomes `..., l_{L-2} → predicts l_{L-1}`, then
`pad → predicts EOS`: the final text token `l_{L-1}` is never an input, and EOS
is conditioned on a pad embedding — unlike inference, where EOS follows the
last generated token. Because the per-speaker CTC labels are split from
`decoder_input_ids` (`models/modeling_speech_encoder_decoder_llama.py:671-691`),
the last speaker's CTC target also silently loses its final token for those
rows. (Verified on dummy data; the same off-by-one existed in the pre-refactor
in-model code.) *Fix:* build `decoder_input_ids = [start] + labels` (full
length L+1, −100→pad); shorter rows are unaffected (their extra positions were
already pad).

**ISSUE-07 — speech-insertion and prompt-split positions are taken from batch sample 0 only.**
`models/modeling_llama.py:136-151,186-191` and `models/modeling_speech_encoder_decoder_llama.py:673-681`. **Major.**
In instruct mode the decoder locates `<bos_speech>`/`<eos_speech>` in
`input_ids[0]` and inserts the speech embeddings at that position **for the
whole batch**; the glue model likewise finds `<bos_response>` in row 0 to slice
the CTC label region. If prompts ever vary in tokenized length within a batch,
every other row is silently mis-aligned (speech inserted in the wrong place,
labels shifted ⇒ corrupted training). Today `utils/generate_dataset.py` uses
one constant prompt for the whole dataset, so this is latent — but nothing
enforces it. Full per-sample insertion would touch the attention/cache plumbing
(architecture-level → *needs author decision*). *Fix (guard):* validate at the
insertion site that all rows share identical `<bos_speech>`/`<eos_speech>`
positions and raise a clear error instead of corrupting silently.

**ISSUE-08 — LoRA merge uses hard-coded `r=16, alpha=32`, ignoring the configurable training values.**
`utils/merge_adapter.py:14,76-81` + `run.sh:246-248`. **Major.**
`run.sh`/`src/arguments.py` expose `selfattn_lora_r/alpha` as flags, but
`merge_adapter.py` always merges with `scaling = 32/16`. Training with any
other r/alpha produces a silently mis-scaled merged `model.safetensors` (wrong
weights at inference). *Fix:* accept `--lora_r/--lora_alpha` CLI args (defaults
preserve old behaviour) and have `run.sh` pass the actual values.

**ISSUE-09 — model-topology flags are inconsistently propagated to inference.**
`inference_asr.py:89-100`, `inference_asr_gpus.py:110-121`, `run.sh` stages 4–6. **Major.**
`finetune_asr.py` copies `talker_ctc_refine`, `r_max`, `lora_alpha` onto the
config (and the saved config retains them), but the inference scripts re-set
only a subset of flags from CLI: the saved value survives only for flags the
script does *not* override, and any overridden flag falls back to its CLI
default when `run.sh` doesn't pass it (stage 5 already omits
`decoder_cross_attention*`). A model trained with `talker_ctc_refine=true` (or
non-default `r_max`) decodes with a silently different topology — mismatched
checkpoint keys are dropped without error. *Fix:* copy the same flag set as
training in both inference scripts and pass `talker_ctc_refine`, `r_max`,
`lora_alpha` through `run.sh` stages 4–6, keeping training and decoding
configurations identical.

**ISSUE-10 — PCGrad block is dead under DDP and overwrites shared gradients.**
`src/trainer_seq2seq.py:1071-1141`. **Major — needs author decision.**
`hasattr(model, "encoder")` is evaluated on the *wrapped* model; under
`torch.distributed.launch` (the standard run.sh path) `model` is
`DistributedDataParallel`, which exposes `.module`, so `shared_params` stays
empty and PCGrad silently never runs in multi-GPU training (single-GPU runs do
use it ⇒ training behaviour differs by world size). Moreover, when it does run,
`p.grad = g` *replaces* the encoder/separator gradients after
`accelerator.backward`, discarding the CE-loss contribution to those modules in
hybrid mode and bypassing DDP gradient averaging. Whether PCGrad should apply
to CTC-head gradients only, and how it should interact with CE/DDP, is an
algorithm decision — not fixed here.

**ISSUE-11 — `ctcprompt` bridge crashes when all CTC heads emit an empty prefix.**
`models/ctc_prompt.py:100-106`. **Major.**
The `if len(all_ids_per_b[b]) == 0: continue` guard is commented out, so
`torch.cat([])` raises `RuntimeError` for any sample where every head produced
only blanks (common early in training with `ctc_bridge_type=ctcprompt`).
*Fix:* restore the guard.

**ISSUE-12 — inference skips the sampling-rate alignment step used in training.**
`inference_asr.py:129`, `inference_asr_gpus.py:150`, `utils/resample_dataset_utils.py`. **Major.**
Training calls `maybe_resample_dataset` (casts the audio column to the feature
extractor's 16 kHz); both inference scripts have it commented out — likely
because the helper assumes a `DatasetDict` (`raw_datasets.values()`) while
inference loads a single `Dataset` split, so the call crashes. With any dataset
not stored at 16 kHz, inference silently feeds wrong-rate audio that training
would have resampled. *Fix:* make the helper handle both `Dataset` and
`DatasetDict`, and re-enable the call in both inference scripts (no-op at
16 kHz).

**ISSUE-13 — `forward_ctc` crashes uninformatively when `talker_ctc=False`.**
`models/modeling_speech_encoder_decoder_llama.py:843-856`. **Minor.**
CTC decoding on a model without CTC heads reaches
`torch.cat(ctc_transcription_list)` with an empty list → opaque
`RuntimeError`. *Fix:* raise a clear `ValueError` explaining that
`--ctc_decoding` requires `talker_ctc=True`.

**ISSUE-14 — `flex_attention` branch references undefined names.**
`models/modeling_llama.py:284-288`. **Minor.**
`make_flex_block_causal_mask` / `BlockMask` are never imported → `NameError`
if anyone loads the model with `attn_implementation="flex_attention"`
(advertised via `_supports_flex_attn = True`). *Fix:* raise
`NotImplementedError` for flex_attention instead of crashing with `NameError`.

**ISSUE-15 — `CTC` (`ctc_type="builtin2"`) references `self.ignore_nan_grad` which is never assigned.**
`models/ctc.py:41-42,73`. **Minor.**
The constructor folds `ignore_nan_grad` into `zero_infinity` but never stores
the attribute, so the `builtin2` path raises `AttributeError`. *Fix:* store the
attribute in `__init__`.

**ISSUE-16 — deprecated `torch.cuda.amp.autocast` API in the loss.**
`models/losses.py:96`. **Minor.**
Emits a `FutureWarning` on every step under torch ≥ 2.4 and will break on
removal. *Fix:* use `torch.amp.autocast("cuda", enabled=False)` guarded for
CPU-only execution.

**ISSUE-17 — eval WER during training counts prompt words as hypothesis text (instruct mode).**
`utils/metric_utils.py:43-52` + `src/trainer_seq2seq.py:625-734`. **Minor — needs author decision.**
With `--predict_with_generate`, `generate()` returns `[bos] + prompt + response`;
`skip_special_tokens=True` removes the special tokens but keeps the prompt
*words* in `pred_str`, while labels have the prompt masked — so the logged
`eval_wer` is inflated. Model selection uses `eval_loss`, so training outcomes
are unaffected. Stripping requires deciding where to split (e.g., on
`<bos_response>` before special-token removal) — deferred to the author.

**ISSUE-18 — magic numbers and prompt-stripping hack in inference.**
`inference_asr.py:183,193,202`, `inference_asr_gpus.py:224,234,243`. **Minor.**
`max_length=150` (can truncate long 2/3-speaker serialized transcripts;
training created the model with `max_length=200`) and
`prompt_ids[1:-4]`-based label cleanup are unexplained constants. Deferred
(changing decode length alters results; author should choose the budget).

**ISSUE-19 — hard-coded cluster paths and dead code.**
`run.sh:3,161,173`, `run_librispeechmix.sh`, `slurm/*` (`/lustre/...` container
images, exclude files, cd paths), `delte_dir.sh` (typo'd name, hard-coded
ROOT), unused `last_ckpt`/`data_collator` in the inference scripts. **Minor.**
Cluster-specific by nature; deferred (documented here). The stage-5/6
`output_dir` hijack from the same family is fixed under ISSUE-02.

**ISSUE-20 — stage 1 writes `datasets/libri{N}mix_noisy` regardless of `corpus`.**
`run.sh:159-166`. **Minor.**
Stage 3 reads `datasets/${corpus}`; stage 1's output name is fixed to
`..._noisy` (with `--suffix ''`), so building e.g. `libri2mix_clean` requires
hand-editing. Choosing the right `suffix`/`wav_scp_name` per corpus is a data
layout decision; deferred (documented).

## Fix plan

1. **ISSUE-01** (Critical) — `remove_unused_columns=False` in `finetune_asr.py`.
2. **ISSUE-02** (Critical) — delete `output_dir` overrides in `run.sh` / `run_librispeechmix.sh`.
3. **ISSUE-03** (Critical) — distributed init before model placement in `inference_asr_gpus.py`.
4. **ISSUE-04** (Critical) — restore missing imports in `src/trainer_seq2seq.py`.
5. **ISSUE-05** (Critical) — pin core dependencies (`transformers==4.53.3`) in `requirements.txt`.
6. **ISSUE-06** (Major) — fix `decoder_input_ids` off-by-one in `src/data_collator.py`.
7. **ISSUE-07** (Major) — batch-uniform prompt guard in `models/modeling_llama.py` (full variable-prompt support: needs author decision).
8. **ISSUE-08** (Major) — parametrize `utils/merge_adapter.py`; pass values from `run.sh`.
9. **ISSUE-09** (Major) — propagate `talker_ctc_refine`/`r_max`/`lora_alpha` to inference configs and run.sh stages 4–6.
10. **ISSUE-11** (Major) — restore empty-prefix guard in `models/ctc_prompt.py`.
11. **ISSUE-12** (Major) — `maybe_resample_dataset` for `Dataset` inputs; re-enable in inference scripts.
12. **ISSUE-13/14/15/16** (Minor, small & safe) — informative `forward_ctc` error; flex-attention guard; `ignore_nan_grad` attr; autocast API.
13. **ISSUE-10** (Major) — needs author decision (documented above).
14. **ISSUE-17/18/19/20** (Minor) — deferred / needs author decision (documented above).
15. Add `tests/test_smoke.py` (label-serialization round-trip, collator alignment, tiny-model forward/loss).

## Fixes applied

Verification after all fixes: `python -m compileall .` exits 0; `finetune_asr`,
`inference_asr`, `inference_asr_gpus` import cleanly under the pinned stack
(`transformers==4.53.3`); `pytest tests/test_smoke.py` → **6 passed**
(label-serialization round-trip, collator teacher-forcing alignment + prompt
masking, tiny-model forward with hybrid CE+CTC loss, greedy `generate()`).

| Issue | Severity | Status | Commit |
|---|---|---|---|
| ISSUE-01 | Critical | fixed | `485cb14` |
| ISSUE-02 | Critical | fixed | `ba67b10` |
| ISSUE-03 | Critical | fixed | `4653954` |
| ISSUE-04 | Critical | fixed | `d42a8e2` |
| ISSUE-05 | Critical | fixed | `57e9ae1` + `2891763` (pin corrected to 4.53.3 after the MRO finding; smoke tests added) |
| ISSUE-06 | Major | fixed | `72334e8` |
| ISSUE-07 | Major | fixed (guard) | `f29d538` — silent corruption replaced by a clear error when prompt lengths differ within a batch; full per-sample speech insertion is an architecture change → **needs author decision** |
| ISSUE-08 | Major | fixed | `fc00814` |
| ISSUE-09 | Major | fixed | `0e63af4` |
| ISSUE-10 | Major | fixed (author decision: opt-in flag) | `9b0dbba` — PCGrad gated behind `--pcgrad` (default **off**, matching all prior multi-GPU runs); shared params resolved on the unwrapped model; skipped with a warning under DDP (`world_size > 1`). |
| ISSUE-11 | Major | fixed | `3c9156a` |
| ISSUE-12 | Major | fixed | `143b767` |
| ISSUE-13 | Minor | fixed | `ba0a7d4` |
| ISSUE-14 | Minor | fixed | `ba0a7d4` |
| ISSUE-15 | Minor | fixed | `ba0a7d4` |
| ISSUE-16 | Minor | fixed | `ba0a7d4` |
| ISSUE-17 | Minor | fixed | `37d8e35` — predictions are cut at `<bos_response>` (token-id level, before decoding) so `eval_wer` compares responses only; no-op for non-instruct tokenizers; `-100` padding in predictions also handled. |
| ISSUE-18 | Minor | fixed | `f9fc1d6` — decode budget is now `--decode_max_length` (default 150 → unchanged results unless overridden); the `prompt_ids[1:-4]` reference-cleanup slice is documented in place. |
| ISSUE-19 | Minor | fixed (partial) | `70a2b35` — `run.sh` accepts `base_data_path=` / `decoder_base=` (defaults preserve the original locations). Paths inside `slurm/*` and `delte_dir.sh` are deployment config and remain deferred. |
| ISSUE-20 | Minor | deferred — stage-1 output naming vs `corpus` requires a data-layout decision (suffix / wav_scp variant per corpus). |

**Entry-point compatibility note:** no script names or CLI flags were removed.
Additions only: `utils/merge_adapter.py` gained optional `--lora_r/--lora_alpha`
(defaults preserve the old behaviour), and `run.sh` stages 4–6 now pass the
already-existing `talker_ctc_refine`/`r_max`/`lora_alpha`/`decoder_cross_attention*`
flags through to inference.
