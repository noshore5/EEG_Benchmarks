"""CPU-only validation that carrying _DenseEdgeMambaContinuous's cache across
chunks/windows actually buys something a per-window-reset model structurally
cannot have -- not just "the plumbing runs without crashing" (already covered
by dense_edge_mamba_continuous_parity.py), but "this paradigm change changes
what's learnable."

Synthetic task, per recording:
  - 3 chunks (chunks == classification windows here, the v1 simplification --
    see this branch's design notes), each [B=1, C_in, E, T_chunk].
  - chunk 0 = noise + (signal_pulse * sign), sign in {-1, +1} chosen once per
    recording. chunks 1 and 2 = PURE noise, zero signal.
  - label for EVERY window in the recording = the recording's sign bit.

A model that only ever sees ONE chunk at a time (state reset every chunk,
reproducing _DenseEdgeMambaTemporal's own per-window-reset behavior) has
LITERALLY ZERO information available for windows 1 and 2 -- their input is
pure noise, nothing correlates with the label. Expected: stuck at chance
(~50%) on those windows, however long trained.

A model that carries _DenseEdgeMambaContinuous's cache forward across chunks
(reset only at recording start) has chunk 0's pulse baked into its state by
the time it processes windows 1/2. Expected: learns to solve them.

Same architecture, same data-generating distribution, same optimizer
settings, same step count for both -- the ONLY difference is whether cache
is carried or reset between chunks. If "continuous" doesn't clearly beat
"reset" on windows 1/2, the mechanism doesn't actually do what this branch
claims.

IMPORTANT: noise for chunks 1/2 is freshly resampled every single step, NOT
a fixed dataset reused across epochs. With a fixed dataset a big enough net
can just MEMORIZE which specific noise pattern maps to which label (table
lookup, achievable by both conditions, proves nothing about cross-window
information use) -- an earlier version of this script had exactly that bug
(both conditions hit 100%). Since chunk 1/2 noise here is never seen twice,
memorization is impossible; only genuine use of carried-forward information
from chunk 0 can push accuracy above chance.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.cwt_gnn_classifiers import (  # noqa: E402
    _DenseEdgeMambaContinuous,
    pool_continuous_edge_stream_to_windows,
)

torch.manual_seed(0)
device = torch.device("cpu")

B, C_IN, E, T_CHUNK = 1, 6, 3, 4
D_MODEL, D_STATE, D_CONV, EXPAND, C_OUT = 8, 8, 3, 2, 8
N_RECORDINGS = 40
N_CHUNKS_PER_RECORDING = 3
PULSE_SCALE = 4.5  # signal amplitude injected only into chunk 0
NOISE_SCALE = 1.0
LR = 1e-3
GRAD_CLIP_NORM = 1.0  # standard for recurrent/SSM training -- without this,
# an early epoch-39 solution destabilized back to chance by epoch 59 (see
# git history of this script for the un-clipped run this replaced).
N_EPOCHS = 80


# Fixed "direction" the sign scales -- this is a learnable, reused pattern
# (like a real recurring EEG motif), NOT the leakage source. The leakage risk
# is per-recording NOISE being reused, which sample_recording avoids by
# drawing fresh torch.randn() every call (no fixed generator/seed captured
# across calls -- see module docstring).
_PULSE_GENERATOR = torch.Generator().manual_seed(12345)
PULSE = torch.randn(C_IN, E, generator=_PULSE_GENERATOR)


def sample_recording() -> tuple[list[torch.Tensor], int]:
    sign = 1 if torch.rand(()).item() < 0.5 else -1
    chunks = []
    for k in range(N_CHUNKS_PER_RECORDING):
        noise = torch.randn(B, C_IN, E, T_CHUNK) * NOISE_SCALE  # fresh every call, never repeated
        if k == 0:
            noise = noise + (PULSE.view(1, C_IN, E, 1) * PULSE_SCALE * sign)
        chunks.append(noise)
    label = 1 if sign == 1 else 0
    return chunks, label


class Head(nn.Module):
    """Toy readout: pooled window [B, C_out, E, 1] -> mean over edges -> linear.
    Deliberately NOT the real sparse_message_mlp/hops/sparse_classifier path
    (already integration-tested separately, see cwt_gnn_classifiers.py's
    pool_continuous_edge_stream_to_windows docstring) -- this script is only
    validating the state-carryover TRAINING MECHANIC, not the full model.
    """

    def __init__(self, c_out: int):
        super().__init__()
        self.fc = nn.Linear(c_out, 2)

    def forward(self, pooled_window: torch.Tensor) -> torch.Tensor:
        # pooled_window: [B, C_out, E, 1]
        feat = pooled_window.squeeze(-1).mean(dim=-1)  # [B, C_out]
        return self.fc(feat)  # [B, 2]


def run_condition(carry_state: bool, seed: int) -> list[float]:
    torch.manual_seed(seed)
    model = _DenseEdgeMambaContinuous(
        in_channels=C_IN, out_channels=C_OUT, d_model=D_MODEL, d_state=D_STATE,
        d_conv=D_CONV, expand=EXPAND, n_layers=1,
    ).to(device)
    head = Head(C_OUT).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    late_window_acc_by_epoch = []
    for epoch in range(N_EPOCHS):
        correct_late, total_late = 0, 0
        for _ in range(N_RECORDINGS):
            chunks, label = sample_recording()  # freshly sampled -- see module docstring
            label_t = torch.tensor([label])
            optimizer.zero_grad()
            cache = None
            for k, chunk in enumerate(chunks):
                out_seq, new_cache = model(chunk, cache=cache)
                cache = new_cache if carry_state else None  # the ONLY difference between conditions
                pooled = pool_continuous_edge_stream_to_windows(out_seq, [(0, out_seq.shape[-1])], pool="last")[0]
                logits = head(pooled)
                loss = loss_fn(logits, label_t)
                loss.backward()
                if k > 0:  # windows 1, 2: the ones with NO signal of their own
                    pred = logits.argmax(dim=-1).item()
                    correct_late += int(pred == label)
                    total_late += 1
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), GRAD_CLIP_NORM
            )
            optimizer.step()
        late_window_acc_by_epoch.append(correct_late / max(1, total_late))
    return late_window_acc_by_epoch


acc_continuous = run_condition(carry_state=True, seed=100)
acc_reset = run_condition(carry_state=False, seed=100)

print("epoch  continuous(carry-state)  reset(per-window)")
for e in (0, 4, 9, 19, 39, 59, 79):
    if e < N_EPOCHS:
        print(f"{e:5d}  {acc_continuous[e]:22.3f}  {acc_reset[e]:17.3f}")

# Best 5-epoch-ROLLING-AVERAGE, not best single epoch or last-N-average:
# this toy loop (40 fresh recordings/epoch, no LR schedule) is noisy
# step-to-step by construction (small-batch, never-repeated data -- see
# module docstring; only 80 late-window samples/epoch, binomial std at
# p=0.5 is ~0.056/epoch), so a single-epoch max is a multiple-testing trap:
# over enough epochs SOME single epoch will spike well above chance by pure
# noise even under the null (confirmed -- an earlier version of this
# assertion using single-epoch max failed on reset hitting 0.675 at one
# noisy epoch while every neighboring epoch sat at 0.40-0.55). Rolling-
# averaging over 5 epochs suppresses that without hiding genuine SUSTAINED
# ability, which is the actual claim under test ("carrying state makes
# this learnable, which resetting cannot do regardless of how long you
# train" -- a sustained-capability question, not a single-lucky-epoch one).
def _rolling_max(values: list[float], window: int = 5) -> float:
    return max(
        sum(values[i : i + window]) / window
        for i in range(len(values) - window + 1)
    )


best_continuous = _rolling_max(acc_continuous)
best_reset = _rolling_max(acc_reset)
final_continuous = sum(acc_continuous[-10:]) / 10
final_reset = sum(acc_reset[-10:]) / 10
print(
    f"\nbest 5-epoch-rolling-avg late-window accuracy: continuous={best_continuous:.3f}  reset={best_reset:.3f}"
)
print(f"final (last-10-epoch avg): continuous={final_continuous:.3f}  reset={final_reset:.3f}")

assert best_continuous > 0.8, f"continuous should clearly solve windows 1/2 at some point in training, got best={best_continuous:.3f}"
assert best_reset < 0.65, f"reset should stay near chance on windows 1/2 (no info available) even at its best epoch, got {best_reset:.3f}"
assert best_continuous > best_reset + 0.15, (
    f"continuous's peak ({best_continuous:.3f}) should clearly beat reset's peak "
    f"({best_reset:.3f}) -- if not, carrying state isn't actually buying anything."
)
print("\nOK: carrying state across chunks solves a task per-window reset structurally cannot.")
