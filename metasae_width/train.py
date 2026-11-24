"""The trainer: one BatchTopK dictionary on unit rows, for a fixed number of epochs.

Every seed is derived from the run id. Extending a run's state to N epochs in all equals training N epochs directly."""

import math
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from metasae_width.metrics import variance_explained
from metasae_width.models import BatchTopKSAE, auxk_loss, fit_threshold_
from metasae_width.seeds import derived_seed, set_numeric_flags

# The recipe: k, epochs, batch cap, dead count in passes over the training rows, AuxK cap.
K, EPOCHS, MAX_BATCH, DEAD_PASSES, AUX_CAP = 4, 2000, 4096, 5, 512
ADAM_BETAS, ADAM_EPS, CLIP_NORM, AUX_WEIGHT = (0.9, 0.999), 1e-8, 1.0, 1 / 32
LOG_EVERY = 100
# Snapshot epochs: every SNAPSHOT_EARLY_EVERY through SNAPSHOT_EARLY_UNTIL, then every SNAPSHOT_EVERY.
SNAPSHOT_EARLY_EVERY, SNAPSHOT_EARLY_UNTIL, SNAPSHOT_EVERY = 25, 250, 250
PARAMS = ("W_enc", "b_enc", "W_dec", "b_dec")


class Schedule(NamedTuple):
    """Batch size, batches per epoch, epochs, total steps, AuxK count, and rows until a silent latent is dead."""

    batch: int
    per_epoch: int
    epochs: int
    steps: int
    k_aux: int
    dead_after: int


def schedule(n_train: int, width: int, epochs: int) -> Schedule:
    batch = min(MAX_BATCH, n_train)
    per_epoch = math.ceil(n_train / batch)
    return Schedule(batch, per_epoch, epochs, epochs * per_epoch, min(AUX_CAP, width // 2), DEAD_PASSES * n_train)


class LogRow(NamedTuple):
    """One logged step: train losses averaged over the last LOG_EVERY steps, validation scores, dead count, norm."""

    step: int
    epoch: int
    train_mse: float
    train_aux: float
    val_mse: float
    val_ve: float
    dead: int
    grad_norm: float


class Trained(NamedTuple):
    """The fitted dictionary on the CPU, the state to continue from, the live mask and the record."""

    sae: BatchTopKSAE
    state: dict[str, Any]
    live: torch.Tensor
    record: dict[str, Any]


def snapshot_epochs(done: int, epochs: int) -> list[int]:
    """The grid epochs e with done <= e < epochs: every SNAPSHOT_EARLY_EVERY through SNAPSHOT_EARLY_UNTIL, then every
    SNAPSHOT_EVERY."""
    early = range(SNAPSHOT_EARLY_EVERY, SNAPSHOT_EARLY_UNTIL + 1, SNAPSHOT_EARLY_EVERY)
    late = range(SNAPSHOT_EARLY_UNTIL + SNAPSHOT_EVERY, epochs, SNAPSHOT_EVERY)
    return [e for e in (*early, *late) if done <= e < epochs]


def update_dead_(counters: torch.Tensor, latents: torch.Tensor, dead_after: int) -> torch.Tensor:
    """Reset the counter of every latent that fired, grow the others by the batch size; return the dead mask."""
    fired = (latents > 0).any(0)
    counters.add_(latents.shape[0]).masked_fill_(fired, 0)
    return counters >= dead_after


def train_step(
    sae: BatchTopKSAE, opt: torch.optim.Optimizer, x: torch.Tensor, counters: torch.Tensor, sched: Schedule
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step; returns the squared error, the AuxK term and the gradient norm before clipping. The decoder rows'
    gradients lose their component along each row, and the rows are unit length again after the update."""
    out = sae(x, "batch_topk")
    dead = update_dead_(counters, out.latents, sched.dead_after)
    residual = x - out.recon
    mse = residual.pow(2).mean()
    aux = auxk_loss(out.pre, residual, sae.W_dec, dead, k_aux=sched.k_aux, weight=AUX_WEIGHT)
    loss = mse + aux
    loss.backward()
    with torch.no_grad():
        unit = F.normalize(sae.W_dec, dim=1)
        sae.W_dec.grad -= (sae.W_dec.grad * unit).sum(dim=1, keepdim=True) * unit  # type: ignore[operator]
    params = [getattr(sae, name) for name in PARAMS]
    total = torch.nn.utils.get_total_norm([p.grad for p in params], 2.0, foreach=False)  # type: ignore[arg-type]
    loss_value, norm_value = torch.stack([loss.detach(), total]).tolist()
    if not (math.isfinite(loss_value) and math.isfinite(norm_value)):
        raise FloatingPointError(f"loss {loss_value}, gradient norm {norm_value}")
    torch.nn.utils.clip_grads_with_norm_(params, CLIP_NORM, total, foreach=False)
    opt.step()
    with torch.no_grad():
        sae.W_dec.copy_(F.normalize(sae.W_dec, dim=1))
    opt.zero_grad(set_to_none=True)
    return mse.detach(), aux.detach(), total.detach()


def adam(sae: BatchTopKSAE, lr: float) -> torch.optim.Adam:
    """Adam over the four parameters, in torch's single-tensor implementation on every device."""
    params = [getattr(sae, name) for name in PARAMS]
    return torch.optim.Adam(
        params, lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0, amsgrad=False, foreach=False, fused=False
    )


def _to_cpu(value: Any) -> Any:
    """A copy of a nested structure with every tensor detached and copied to the CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


@torch.no_grad()
def _score(sae: BatchTopKSAE, rows: torch.Tensor) -> tuple[float, float]:
    """Mean squared error and variance explained of ``rows`` presented as one batch."""
    recon = sae(rows, "batch_topk").recon
    return float((recon - rows).pow(2).mean()), variance_explained(recon, rows)


def _start(
    rows: torch.Tensor, run_id: str, width: int, lr: float, device: Any, state: dict[str, Any] | None
) -> tuple[BatchTopKSAE, torch.optim.Adam, torch.Tensor, int]:
    """The model, Adam, dead counters and epochs done: new (unit-norm Kaiming-uniform decoder rows from a CPU generator
    seeded by ``<run_id>/init``, the encoder their transpose, b_dec the rows' mean), or restored from ``state``."""
    sae = BatchTopKSAE(rows.shape[1], width, K)
    if state is None:
        g = torch.Generator().manual_seed(derived_seed(f"{run_id}/init"))
        empty = torch.empty(width, sae.dim, dtype=torch.float32)
        w = F.normalize(torch.nn.init.kaiming_uniform_(empty, a=math.sqrt(5), generator=g), dim=1)
        with torch.no_grad():
            sae.W_dec.copy_(w)
            sae.W_enc.copy_(w.T)
            sae.b_dec.copy_(rows.detach().to("cpu", torch.float64).mean(0).to(torch.float32))
        sae.to(device)
        counters = torch.zeros(width, dtype=torch.int64, device=device)
        return sae, adam(sae, lr), counters, 0
    sae.load_state_dict(state["weights"], strict=False)  # the threshold is refitted at the end
    sae.to(device)
    opt = adam(sae, lr)
    opt.load_state_dict(_to_cpu(state["adam"]))
    return sae, opt, state["counters"].to(device, copy=True), state["epochs_done"]


def _steps(
    sae: BatchTopKSAE,
    opt: torch.optim.Adam,
    counters: torch.Tensor,
    stored: torch.Tensor,
    sched: Schedule,
    run_id: str,
    done: int,
) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run the epochs after the ``done`` first, yielding (step, epoch, mse, aux, norm) after each step."""
    step = done * sched.per_epoch
    for epoch in range(done, sched.epochs):
        seed = derived_seed(f"{run_id}/shuffle/{epoch}")
        perm = torch.randperm(len(stored), generator=torch.Generator().manual_seed(seed)).to(stored.device)
        for start in range(0, len(stored), sched.batch):
            mse, aux, norm = train_step(sae, opt, stored[perm[start : start + sched.batch]], counters, sched)
            step += 1
            yield step, epoch, mse, aux, norm


def _finish(
    sae: BatchTopKSAE, stored: torch.Tensor, sched: Schedule, held: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit the threshold over the training rows in training-sized batches, find the live latents in the same
    pass, and score the held-out rows. Returns the live mask and the measured part of the record."""
    fitting = [stored[start : start + sched.batch] for start in range(0, len(stored), sched.batch)]
    threshold = fit_threshold_(sae, fitting)
    with torch.no_grad():
        above = (sae.select(sae.preact(batch), "threshold") for batch in fitting)
        live = torch.stack([(batch > 0).any(dim=0).cpu() for batch in above]).any(dim=0)
    final: dict[str, float] = {}
    for name, rows in held.items():
        final[f"{name}_mse"], final[f"{name}_ve"] = _score(sae, rows)
    dead_fraction = int((~live).sum().item()) / live.numel()
    record = {"schedule": sched._asdict(), "threshold": threshold, "dead_fraction": dead_fraction}
    return live, record | {"n_live": int(live.sum()), "final": final}


def _state(
    sae: BatchTopKSAE, opt: torch.optim.Adam, counters: torch.Tensor, *, run_id: str, width: int, lr: float, epochs: int
) -> dict[str, Any]:
    """What a later call needs to continue the run."""
    return {
        "run_id": run_id,
        "width": width,
        "lr": lr,
        "weights": {name: _to_cpu(getattr(sae, name)) for name in PARAMS},
        "adam": _to_cpu(opt.state_dict()),
        "counters": _to_cpu(counters),
        "epochs_done": epochs,
    }


def train_on_rows(
    rows: torch.Tensor,
    val: torch.Tensor,
    test: torch.Tensor,
    *,
    run_id: str,
    width: int,
    lr: float,
    epochs: int,
    device: str | torch.device,
    state: dict[str, Any] | None = None,
    on_log: Callable[[LogRow], None] = lambda row: None,
    on_snapshot: Callable[[int, dict[str, torch.Tensor]], None] = lambda step, weights: None,
) -> Trained:
    """Train on ``rows``, epoch e in the order of a randperm seeded by ``<run_id>/shuffle/<e>``; with ``state``,
    continue that run to ``epochs`` in all. ``on_snapshot`` gets the step and CPU copies of the four parameters at
    the end of every snapshot_epochs epoch this call covers, including one that is the epoch it starts from."""
    set_numeric_flags()
    sched = schedule(len(rows), width, epochs)
    sae, opt, counters, done = _start(rows, run_id, width, lr, device, state)
    recent_mse, recent_aux = deque(maxlen=LOG_EVERY), deque(maxlen=LOG_EVERY)
    stored, val_set, test_set = rows.to(device), val.to(device), test.to(device)
    wanted = {e * sched.per_epoch for e in snapshot_epochs(done, epochs)}

    def snapshot(step: int) -> None:
        on_snapshot(step, {name: _to_cpu(getattr(sae, name)) for name in PARAMS})

    if done * sched.per_epoch in wanted:
        snapshot(done * sched.per_epoch)
    for step, epoch, mse, aux, norm in _steps(sae, opt, counters, stored, sched, run_id, done):
        if step in wanted:
            snapshot(step)
        recent_mse.append(mse)
        recent_aux.append(aux)
        if step % LOG_EVERY == 0 or step == sched.steps:
            val_mse, val_ve = _score(sae, val_set)
            dead = int((counters >= sched.dead_after).sum())
            train_mse, train_aux = (float(torch.stack(list(w)).mean()) for w in (recent_mse, recent_aux))
            on_log(LogRow(step, epoch, train_mse, train_aux, val_mse, val_ve, dead, float(norm)))
    live, record = _finish(sae, stored, sched, {"val": val_set, "test": test_set})
    new_state = _state(sae, opt, counters, run_id=run_id, width=width, lr=lr, epochs=epochs)
    return Trained(sae.cpu(), new_state, live, {**record, "snapshot_steps": sorted(wanted)})
