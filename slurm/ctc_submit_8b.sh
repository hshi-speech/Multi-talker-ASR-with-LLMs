#!/usr/bin/env bash
# Converted from a SLURM submitter to a plain shell script (no sbatch).
# Each configuration below now runs SEQUENTIALLY on this machine via run_job.sh;
# a failing configuration is reported but does not stop the remaining ones.
# Limit GPUs with CUDA_VISIBLE_DEVICES if needed.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
FAILED_JOBS=()

ctc=true

adapter_only_decoder=false
train_mode=ctc
ef=true

partial_encoder_unfreeze=""
partial_decoder_unfreeze=""
partial_others_unfreeze="separator,serialized_ctc"


per_device_train_batch_size=8
per_device_eval_batch_size=8

stage=5
stop_stage=5


# -----------------------> two talker condition
tn=2

dec=Meta-Llama-3.1-8B
corp=libri2mix_noisy
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
pt_separator=exp_separator/libri2mix_noisy_llama-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_noisy
pt_separator=exp_separator/libri2mix_noisy_llama-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B
corp=libri2mix_clean
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_clean
pt_separator=exp_separator/libri2mix_clean_llama-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_clean
pt_separator=exp_separator/libri2mix_clean_llama-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

# -----------------------> three talker condition
tn=3

dec=Meta-Llama-3.1-8B
corp=libri3mix_noisy
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_noisy
pt_separator=exp_separator/libri3mix_noisy_llama-ins-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_noisy
pt_separator=exp_separator/libri3mix_noisy_llama-ins-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B
corp=libri3mix_clean
ins=false
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_clean
pt_separator=exp_separator/libri3mix_clean_llama-ins-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("$dec-$corp-$ins")

dec=Meta-Llama-3.1-8B-Instruct
ins=true
pmp=exp_finished/wavlm-Meta-Llama-3.1-8B-Instruct-encoder_unfreeze-decoder_freeze-adater_decoder-libri3mix_clean
pt_separator=exp_separator/libri3mix_clean_llama-ins-1b.pt
echo "[job] $dec-$corp-$ins"
bash run_job.sh \
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
