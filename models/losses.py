# losses.py
import torch
import torch.nn as nn
import os


# NOTE: PIT (Permutation Invariant Training) infrastructure was removed because it had been
# unreachable across multiple layers:
#   - `do_pit = False` was hardcoded inside HybridLoss.forward
#   - the `use_pit` constructor flag was never set to True by the caller
#   - `self.perm_mode = None` was hardcoded, so build_perm always returned the identity perm
# Helpers `build_perm`, `batch_swap_stats`, and `pit_ctc_loss` were defined but never called
# from any active code path. The CTC supervision now uses fixed-order assignment
# (head i ↔ speaker i in the dataset's split order).
class HybridLoss(nn.Module):
    """
    A flexible loss module that can compute:
    - Attention loss only
    - Serialized CTC loss only
    - Hybrid loss (weighted combination of attention + CTC)
    """
    def __init__(self, alpha: float = 0.7, mode: str = 'hybrid',
                 blank_id: int | None = None,
                 enable_blank_check: bool = False,
                 log_every_steps: int = 0):
        super().__init__()
        assert mode in ('attention', 'ctc', 'hybrid'), "mode must be 'attention', 'ctc', or 'hybrid'"
        self.alpha = alpha
        self.mode = mode
        self.ce_loss = nn.CrossEntropyLoss()

        # checks / debug
        self.blank_id = blank_id
        self.enable_blank_check = enable_blank_check
        self.log_every_steps = int(log_every_steps)
        self.log_dict = {}

    def forward(
        self,
        decoder_outputs=None,
        labels=None,
        decoder_vocab_size=None,
        talker_ctc=None,
        sep_hidden_states=None,
        encoder_attention_mask_ctc=None,
        label_spks=None,
        label_spks_lengths=None,
        talker_numbers=1,
        return_dict=True,
    ):
        loss_attn = 0.0
        loss_ctc = 0.0

        # -----------------------------
        # Attention loss (if needed)
        # -----------------------------
        if self.mode in ('attention', 'hybrid'):
            if decoder_outputs is None or labels is None or decoder_vocab_size is None:
                raise ValueError("decoder_outputs, labels, decoder_vocab_size must be provided for attention loss")
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_attn = self.ce_loss(logits.reshape(-1, decoder_vocab_size), labels.reshape(-1))

        # -----------------------------
        # CTC loss (if needed)
        # -----------------------------
        if self.mode in ('ctc', 'hybrid'):
            if (talker_ctc is None or sep_hidden_states is None or encoder_attention_mask_ctc is None
                or label_spks is None or label_spks_lengths is None):
                raise ValueError("CTC related inputs must be provided for CTC loss")

            N = int(talker_numbers)
            assert len(talker_ctc) == N, f"len(talker_ctc)={len(talker_ctc)} != talker_numbers={N}"
            assert len(sep_hidden_states) == len(label_spks) == len(label_spks_lengths) == N, \
                "Mismatch among heads/labels/lengths"

            hlens = encoder_attention_mask_ctc.sum(dim=1).long()  # (B,)
            B = hlens.size(0)
            for i in range(N):
                x, y, yl = sep_hidden_states[i], label_spks[i], label_spks_lengths[i]
                assert x.size(0) == y.size(0) == yl.size(0) == B, f"batch dim mismatch @head {i}"
                assert yl.dtype in (torch.int32, torch.int64), f"length dtype must be int @head {i}"

            # Optional blank-range check
            step = int(getattr(self, "global_step", 0))
            if self.enable_blank_check and (self.blank_id is not None) and (step % max(1, self.log_every_steps or 1000) == 0):
                with torch.no_grad():
                    for i in range(N):
                        if label_spks_lengths[i].sum().item() > 0:
                            max_id = int(label_spks[i].max().item())
                            assert max_id < self.blank_id, \
                                f"[CTC blank check] head {i}: target id {max_id} >= blank_id {self.blank_id}"

            # Fixed-order CTC supervision: head i ↔ speaker i in the dataset's split order.
            # (PIT and the static perm modes were removed — see top-of-file note.)
            ctc_per_head = []
            with torch.cuda.amp.autocast(enabled=False):
                for i, ctc_head in enumerate(talker_ctc):
                    li = ctc_head(
                        sep_hidden_states[i].float(),
                        hlens,
                        label_spks[i],
                        label_spks_lengths[i],
                    )
                    if li.dim() == 0:
                        # fallback: make it per-sample for consistency
                        li = li.unsqueeze(0).expand(B)
                    elif li.dim() > 1:
                        li = li.reshape(-1)
                    ctc_per_head.append(li)  # each (B,)
            loss_ctc = torch.stack([l.mean() for l in ctc_per_head]).mean()

            """
            # -------- debug: grad conflict on shared params --------
            if shared_params is not None and torch.is_grad_enabled():
                shared_params = [p for p in shared_params if p.requires_grad]
                grads = []
                for Li in ctc_per_head:
                    Li_use = Li.mean() if Li.dim() > 0 else Li
                    gi = torch.autograd.grad(
                        Li_use, shared_params,
                        retain_graph=True,
                        allow_unused=True
                    )
                    gi = [g if g is not None else torch.zeros_like(p)
                          for g, p in zip(gi, shared_params)]
                    grads.append(gi)
                # ...(your cosine debug stays same)
                # -------- debug: grad conflict on shared params --------
                def flat(glist):
                    return torch.cat([g.reshape(-1) for g in glist])

                gflat = [flat(g) for g in grads]

                cos_mat = [[0.0] * N for _ in range(N)]
                conflict_cnt, total_cnt = 0, 0
                min_cos, min_pair = 1.0, None

                for i in range(N):
                        for j in range(i + 1, N):
                            cos_ij = torch.nn.functional.cosine_similarity(
                                gflat[i], gflat[j], dim=0, eps=1e-12
                            ).item()
                            cos_mat[i][j] = cos_ij
                            total_cnt += 1
                            if cos_ij < 0:
                                conflict_cnt += 1
                            if cos_ij < min_cos:
                                min_cos = cos_ij
                                min_pair = (i, j)

                if int(os.environ.get("RANK", "0")) == 0:
                        print("perm_used (as best_perm):", perm)
                        print("grad_cosine_matrix(upper):", cos_mat)
                        print("conflict_rate:", conflict_cnt / max(1, total_cnt))
                        print("worst_conflict_pair:", min_pair, "cos=", min_cos)
                # --------------------------------------
            """

        # -----------------------------
        # Combine (return scalar!)
        # -----------------------------
        if self.mode == 'attention':
            total_loss = loss_attn
            self.last_ctc_per_head = None
        elif self.mode == 'ctc':
            total_loss = loss_ctc
            self.last_ctc_per_head = ctc_per_head
        elif self.mode == 'hybrid':
            total_loss = self.alpha * loss_attn + (1.0 - self.alpha) * loss_ctc
            self.last_ctc_per_head = ctc_per_head

        return total_loss

