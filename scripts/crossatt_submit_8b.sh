#!/usr/bin/env bash
# Converted from a SLURM submitter to a plain shell script (no sbatch).
# Each configuration below runs SEQUENTIALLY on this machine, calling ../run.sh
# directly (defaults for unlisted flags live in run.sh itself);
# a failing configuration is reported but does not stop the remaining ones.
# Limit GPUs with CUDA_VISIBLE_DEVICES if needed.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
FAILED_JOBS=()

ctc=true

adapter_only_decoder=false
train_mode=attention
ef=true

partial_encoder_unfreeze=""
partial_decoder_unfreeze=""
partial_others_unfreeze="cross_att_adap"

decoder_cross_attention=true
decoder_cross_attention_type=tiny
decoder_cross_attention_feature=sep

per_device_train_batch_size=8
per_device_eval_batch_size=8

stage=3
stop_stage=4


# -----------------------> two talker condition
tn=2

dec=Meta-Llama-3.1-8B
corp=libri2mix_noisy
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_noisy
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_noisy
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B
corp=libri2mix_clean
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_clean
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri2mix_clean
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

# -----------------------> three talker condition
tn=3

dec=Meta-Llama-3.1-8B
corp=libri3mix_noisy
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_noisy
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_noisy
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B
corp=libri3mix_clean
ins=false
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_clean
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_ctc_finished/mode_ctc-wavlm-Meta-Llama-3.1-8B-Instruct-encoder_freeze-decoder_freeze-adater_encoder_decoder-ctc-libri3mix_clean
echo "[job] $dec-$corp-$ins"
bash ../run.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")



if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
  echo "[WARN] failed configurations:"
  printf '  - %s\n' "${FAILED_JOBS[@]}"
  exit 1
fi
echo "[done] all configurations finished."
