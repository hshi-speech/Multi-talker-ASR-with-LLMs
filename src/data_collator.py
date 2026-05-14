# Create by Hao at 2025-07-01
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    decoder_end_token_id: int
    config: Any

    # filled in __post_init__
    forward_attention_mask: bool = field(init=False)

    def __post_init__(self):
        self.forward_attention_mask = (
            getattr(self.config, "model_type", None) == "whisper"
            and getattr(self.config, "apply_spec_augment", False)
            and getattr(self.config, "mask_time_prob", 0) > 0
        )

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        ignore_id = self.config.ignore_token_id
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), ignore_id)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # --- Build decoder_input_ids and insert EOS / mask prompt (was: in model forward) ---
        # Previously, the model forward at ~line 670-742 of
        # modeling_speech_encoder_decoder_llama.py performed:
        #   (a) shift_tokens_right(labels) -> decoder_input_ids
        #   (b) appended a padding column to make room for EOS
        #   (c) overwrote the first -100 with eos_token_id
        #   (d) masked the prompt prefix using labels[0]-only positions (BUGGY for variable prompts).
        # We move (a)-(c) here and replace (d) with a correct per-sample mask, so the model only
        # has to prepend [-100] * speech_len to align with the post-speech-insertion logits.
        B = labels.shape[0]
        pad_token_id = self.processor.tokenizer.pad_token_id

        # (a) decoder_input_ids = [<dec_start>, labels[:, :-1]], with -100 mapped to pad
        decoder_input_ids = torch.full_like(labels, pad_token_id)
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        decoder_input_ids[:, 1:] = labels[:, :-1]
        decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == ignore_id, pad_token_id)

        # (b) append a column to make room for EOS
        decoder_input_ids = torch.cat(
            [decoder_input_ids, torch.full((B, 1), pad_token_id, dtype=decoder_input_ids.dtype)],
            dim=1,
        )
        labels = torch.cat(
            [labels, torch.full((B, 1), ignore_id, dtype=labels.dtype)],
            dim=1,
        )

        # (c) insert eos at first ignore_id (= right after the transcription, BEFORE we apply the
        # prompt-mask, so this still finds the post-transcription pad region correctly).
        end_id = self.decoder_end_token_id
        if isinstance(end_id, (list, tuple)):
            end_id = end_id[0]
        first_pad_id = (labels == ignore_id).float().argmax(dim=1)
        labels[torch.arange(B), first_pad_id] = end_id

        # (d) per-sample prompt masking — replaces the buggy in-model labels[0]-only logic
        prompt_lens = torch.tensor(
            [feature["prompt_token_len"] for feature in features], dtype=torch.long
        )
        if prompt_lens.max().item() > 0:
            seq_idx = torch.arange(labels.shape[1]).unsqueeze(0)
            prompt_mask = seq_idx < prompt_lens.unsqueeze(1)
            labels = labels.masked_fill(prompt_mask, ignore_id)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        # The self.processor.tokenizer.pad should use "input_ids" as input feature
        # Thus, we put the feature["prompt_ids"] as "input_ids"
        prompts_features= [{"input_ids": feature["prompt_ids"]} for feature in features]
        prompts_batch = self.processor.tokenizer.pad(prompts_features, return_tensors="pt")
        batch["prompt_ids"] = prompts_batch["input_ids"]

        return batch

