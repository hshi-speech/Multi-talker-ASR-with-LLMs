#!/usr/bin/env bash
# Created by Hao at 2026-06-12
# Plain-shell replacement for template.slurm — no SLURM, no container.
# Runs ONE configuration directly on the local machine (all visible GPUs).
#
# Usage (same key=value interface template.slurm had):
#   bash run_job.sh decoder=Llama-3.2-1B corpus=libri2mix_noisy instruct=false ...
#
# Limit GPUs with CUDA_VISIBLE_DEVICES, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 bash run_job.sh stage=3 stop_stage=3 ...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[run_job] args: $@"

########################################
# 1) Parse script args: key=value
########################################
for arg in "$@"; do
  case "$arg" in
    stage=*)                          stage="${arg#*=}" ;;
    stop_stage=*)                     stop_stage="${arg#*=}" ;;
    epoch=*)                          epoch="${arg#*=}" ;;
    encoder=*)                        encoder="${arg#*=}" ;;
    decoder=*)                        decoder="${arg#*=}" ;;
    corpus=*)                         corpus="${arg#*=}" ;;
    instruct=*)                       instruct="${arg#*=}" ;;
    talker_ctc=*)                     talker_ctc="${arg#*=}" ;;
    talker_ctc_refine=*)              talker_ctc_refine="${arg#*=}" ;;
    talker_numbers=*)                 talker_numbers="${arg#*=}" ;;
    separator_hidden=*)               separator_hidden="${arg#*=}" ;;
    train_mode=*)                     train_mode="${arg#*=}" ;;
    encoder_freeze=*)                 encoder_freeze="${arg#*=}" ;;
    decoder_freeze=*)                 decoder_freeze="${arg#*=}" ;;
    decoder_cross_attention=*)        decoder_cross_attention="${arg#*=}" ;;
    decoder_cross_attention_type=*)   decoder_cross_attention_type="${arg#*=}" ;;
    decoder_cross_attention_feature=*) decoder_cross_attention_feature="${arg#*=}" ;;
    adapter_only_decoder=*)           adapter_only_decoder="${arg#*=}" ;;
    per_device_train_batch_size=*)    per_device_train_batch_size="${arg#*=}" ;;
    per_device_eval_batch_size=*)     per_device_eval_batch_size="${arg#*=}" ;;
    partial_encoder_unfreeze=*)       partial_encoder_unfreeze="${arg#*=}" ;;
    partial_decoder_unfreeze=*)       partial_decoder_unfreeze="${arg#*=}" ;;
    partial_others_unfreeze=*)        partial_others_unfreeze="${arg#*=}" ;;
    pretrain_model_path=*)            pretrain_model_path="${arg#*=}" ;;
    pretrain_separator_path=*)        pretrain_separator_path="${arg#*=}" ;;
    precision=*)                      precision="${arg#*=}" ;;
    output_dir=*)                     output_dir="${arg#*=}" ;;
    eval_steps=*)                     eval_steps="${arg#*=}" ;;
    virtual_env=*)                    virtual_env="${arg#*=}" ;;
    cache_dir=*)                      cache_dir="${arg#*=}" ;;
    r_max=*)                          r_max="${arg#*=}" ;;
    lora_alpha=*)                     lora_alpha="${arg#*=}" ;;
    seed=*)                           seed="${arg#*=}" ;;
    ctc_bridge=*)                     ctc_bridge="${arg#*=}" ;;
    ctc_bridge_type=*)                ctc_bridge_type="${arg#*=}" ;;
    pcgrad=*)                         pcgrad="${arg#*=}" ;;
    base_data_path=*)                 base_data_path="${arg#*=}" ;;
    decoder_base=*)                   decoder_base="${arg#*=}" ;;
    *) echo "[run_job] WARN: unknown arg: $arg" >&2 ;;
  esac
done

########################################
# 2) Defaults — single '-' respects explicitly-passed empty strings
########################################
# Fixed/common parameters
stage="${stage-3}"
stop_stage="${stop_stage-3}"
epoch="${epoch-50}"
encoder="${encoder-wavlm}"
eval_steps="${eval_steps-1600}"
# Empty virtual_env -> run.sh uses the python on PATH
virtual_env="${virtual_env-}"
cache_dir="${cache_dir-$HOME/.hf_cache}"

# Sweep parameters
decoder="${decoder-Llama-3.2-1B}"
corpus="${corpus-libri3mix_clean}"
instruct="${instruct-false}"
talker_ctc="${talker_ctc-true}"
talker_ctc_refine="${talker_ctc_refine-false}"
talker_numbers="${talker_numbers-3}"
separator_hidden="${separator_hidden-796}"
decoder_cross_attention="${decoder_cross_attention-false}"
decoder_cross_attention_type="${decoder_cross_attention_type-tiny}"
decoder_cross_attention_feature="${decoder_cross_attention_feature-raw}"
r_max="${r_max-8}"
lora_alpha="${lora_alpha-8}"

# Other toggles
train_mode="${train_mode-hybrid}"
encoder_freeze="${encoder_freeze-false}"
decoder_freeze="${decoder_freeze-true}"
adapter_only_decoder="${adapter_only_decoder-true}"
precision="${precision:-fp32}"
ctc_bridge="${ctc_bridge:-false}"
ctc_bridge_type="${ctc_bridge_type:-raw}"
pcgrad="${pcgrad-false}"

# Batch sizes
per_device_train_batch_size="${per_device_train_batch_size-16}"
per_device_eval_batch_size="${per_device_eval_batch_size-16}"

# IMPORTANT: allow empty string to pass through
partial_encoder_unfreeze="${partial_encoder_unfreeze-}"                   # keep "" if provided
partial_decoder_unfreeze="${partial_decoder_unfreeze-}"                   # keep "" if provided
partial_others_unfreeze="${partial_others_unfreeze-enc_to_dec_proj}"      # only if UNSET, not empty
pretrain_model_path="${pretrain_model_path-}"                             # keep "" if provided
seed="${seed-42}"
pretrain_separator_path="${pretrain_separator_path:-none}"

# Overridable base paths (run.sh has the original defaults; pass only if set)
base_data_path="${base_data_path-}"
decoder_base="${decoder_base-}"
EXTRA_PATH_ARGS=()
[ -n "$base_data_path" ] && EXTRA_PATH_ARGS+=("base_data_path=$base_data_path")
[ -n "$decoder_base" ]   && EXTRA_PATH_ARGS+=("decoder_base=$decoder_base")

########################################
# 3) Run directly (was: inside the SLURM container)
########################################
output_dir="${output_dir-exp}"   # root; run.sh appends the flag-derived tag

bash ../run.sh \
  stage=$stage \
  stop_stage=$stop_stage \
  epoch=$epoch \
  corpus=$corpus \
  encoder=$encoder \
  decoder=$decoder \
  encoder_freeze=$encoder_freeze \
  decoder_freeze=$decoder_freeze \
  per_device_train_batch_size=$per_device_train_batch_size \
  per_device_eval_batch_size=$per_device_eval_batch_size \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  pretrain_model_path="${pretrain_model_path}" \
  adapter_only_decoder=$adapter_only_decoder \
  train_mode=$train_mode \
  instruct=$instruct \
  talker_ctc=$talker_ctc \
  talker_ctc_refine=$talker_ctc_refine \
  precision=$precision \
  separator_hidden=$separator_hidden \
  pretrain_separator_path=$pretrain_separator_path \
  decoder_cross_attention=$decoder_cross_attention \
  decoder_cross_attention_type=$decoder_cross_attention_type \
  decoder_cross_attention_feature=$decoder_cross_attention_feature \
  r_max=${r_max} \
  lora_alpha=${lora_alpha} \
  cache_dir=$cache_dir \
  seed=$seed \
  talker_numbers=$talker_numbers \
  ctc_bridge=$ctc_bridge \
  ctc_bridge_type=$ctc_bridge_type \
  pcgrad=$pcgrad \
  output_dir="$output_dir" \
  eval_steps=$eval_steps \
  virtual_env="$virtual_env" \
  ${EXTRA_PATH_ARGS[@]+"${EXTRA_PATH_ARGS[@]}"}
