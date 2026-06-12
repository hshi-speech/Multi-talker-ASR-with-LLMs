#!/usr/bin/env bash
# Converted from a SLURM submitter to a plain shell script (no sbatch).
# Each configuration below runs SEQUENTIALLY on this machine, calling ../run.sh
# directly (defaults for unlisted flags live in run.sh itself);
# a failing configuration is reported but does not stop the remaining ones.
# Limit GPUs with CUDA_VISIBLE_DEVICES if needed.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
FAILED_JOBS=()

ctc=false
train_mode=attention # mode can be choose from ctc/hybrid/attention
ef=false
adapter_only_decoder=true

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

per_device_train_batch_size=12
per_device_eval_batch_size=12

stage=4
stop_stage=4

# -----------------------> two talker condition
tn=2

corp=libri2mix_noisy
dec=Llama-3.2-1B
ins=false
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn"  encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")

dec=Llama-3.2-1B-Instruct
ins=true
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")

corp=libri2mix_clean
dec=Llama-3.2-1B
ins=false
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")


dec=Llama-3.2-1B-Instruct
ins=true
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")

# -----------------------> three talker condition
tn=3

dec=Llama-3.2-1B
corp=libri3mix_noisy
ins=false
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")

dec=Llama-3.2-1B-Instruct
ins=true
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")


dec=Llama-3.2-1B
corp=libri3mix_clean
ins=false
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")

dec=Llama-3.2-1B-Instruct
ins=true
echo "[job] ${dec}-${corp}-${ins}"
bash ../run.sh \
  decoder="$dec" corpus="$corp" instruct="$ins" talker_ctc="$ctc" talker_numbers="$tn" encoder_freeze="${ef:-}" train_mode="${train_mode:-}" adapter_only_decoder="${adapter_only_decoder:-}" stage="${stage:-}" stop_stage="${stop_stage:-}" per_device_train_batch_size="$per_device_train_batch_size" per_device_eval_batch_size="$per_device_eval_batch_size" \
  partial_encoder_unfreeze="$partial_encoder_unfreeze" \
  partial_decoder_unfreeze="$partial_decoder_unfreeze" \
  partial_others_unfreeze="$partial_others_unfreeze" \
  || FAILED_JOBS+=("${dec}-${corp}-${ins}")



if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
  echo "[WARN] failed configurations:"
  printf '  - %s\n' "${FAILED_JOBS[@]}"
  exit 1
fi
echo "[done] all configurations finished."
