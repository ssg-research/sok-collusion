"""Batched CIFAR Geiping gradient inversion.

Per-record gradient inversion optimizes one 32x32 image against a single-sample
init-model gradient. Each step needs a second-order derivative (backprop through a
per-sample parameter gradient), which is compute-heavy at batch=1 and barely uses an
A100; running many single-image processes oversubscribes the GPU (measured ~13x
per-worker slowdown at 32 procs).

This module reconstructs a whole BATCH of B images simultaneously: per-sample
parameter gradients are computed with torch.func.vmap(grad(...)), so one optimizer
step matches B images against their B target gradients using the GPU's batch
parallelism. The per-image objective (init model, cosine-gradient loss, TV prior,
[-1,1] clamp) is the standard Geiping inversion, so the recons are drop-in for
build_gifd_pool_image.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torchvision.utils import save_image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from amulet.utils import initialize_model
from amulet.datasets import load_cifar10, load_cifar100
from common import compute_member_indices


def tv_loss_batch(x):  # x: (B,3,H,W) -> scalar (mean over batch)
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
           (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def build_model(target_seed: int, num_classes: int, device: str):
    import logging
    log = logging.getLogger("cifar_recon_batched")
    if not log.hasHandlers():
        log.addHandler(logging.StreamHandler())
    torch.manual_seed(target_seed)
    model = initialize_model("resnet", "m1", num_features=3 * 32 * 32,
                             num_classes=num_classes, log=log).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def make_grad_fns(model):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_on_sample(p, x, y):
        out = functional_call(model, (p, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    per_sample_grad = grad(loss_on_sample, argnums=0)            # d loss / d params, for one (x,y)
    batched_grad = vmap(per_sample_grad, in_dims=(None, 0, 0))   # over a batch of (x,y)
    return params, buffers, batched_grad


def batched_cosine_loss(dummy: dict, target: dict, B: int):
    """1 - cosine(dummy_i, target_i) averaged over the batch. Grads concatenated
    across all params per sample (standard Geiping cosine-gradient loss)."""
    dot = torch.zeros(B, device=next(iter(dummy.values())).device)
    pn0 = torch.zeros_like(dot)
    pn1 = torch.zeros_like(dot)
    for k in dummy:
        dg = dummy[k].reshape(B, -1)
        tg = target[k].reshape(B, -1)
        dot = dot + (dg * tg).sum(1)
        pn0 = pn0 + dg.pow(2).sum(1)
        pn1 = pn1 + tg.pow(2).sum(1)
    return (1.0 - dot / (pn0.sqrt() * pn1.sqrt() + 1e-12)).mean()


def invert_batch_robust(model, params, buffers, batched_grad, x_targets, y_targets,
                        *, iters, lr, tv_weight, device, max_retries=20):
    """OOM-resilient wrapper for a shared GPU: on CUDA OOM, free cache and wait for
    other users' memory to drain, retrying the same batch; if OOM persists, split the
    batch in half and recurse so progress continues under heavy contention."""
    for attempt in range(max_retries):
        try:
            return invert_batch(model, params, buffers, batched_grad,
                                x_targets, y_targets, iters=iters, lr=lr,
                                tv_weight=tv_weight, device=device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt < max_retries - 1:
                time.sleep(30)  # let co-tenants' jobs free memory
                continue
            # last resort: split the batch
            B = x_targets.shape[0]
            if B == 1:
                raise
            h = B // 2
            torch.cuda.empty_cache()
            left = invert_batch_robust(model, params, buffers, batched_grad,
                                       x_targets[:h], y_targets[:h], iters=iters, lr=lr,
                                       tv_weight=tv_weight, device=device, max_retries=max_retries)
            right = invert_batch_robust(model, params, buffers, batched_grad,
                                        x_targets[h:], y_targets[h:], iters=iters, lr=lr,
                                        tv_weight=tv_weight, device=device, max_retries=max_retries)
            return torch.cat([left, right], dim=0)


def invert_batch(model, params, buffers, batched_grad, x_targets, y_targets,
                 *, iters, lr, tv_weight, device):
    """x_targets: (B,3,32,32) in [-1,1]. Returns recovered (B,3,32,32) in [-1,1]."""
    B = x_targets.shape[0]
    with torch.no_grad():
        target_grads = batched_grad(params, x_targets, y_targets)
        target_grads = {k: v.detach() for k, v in target_grads.items()}

    x = (torch.randn(B, 3, 32, 32, device=device) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for step in range(iters):
        opt.zero_grad()
        dummy = batched_grad(params, x, y_targets)
        loss = batched_cosine_loss(dummy, target_grads, B) + tv_weight * tv_loss_batch(x)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
    return x.detach()


def load_cifar(dataset: str, root: Path):
    if dataset == "cifar10":
        data = load_cifar10(root / "data" / "cifar10"); num_classes = 10
    else:
        data = load_cifar100(root / "data" / "cifar100"); num_classes = 100
    train = data.train_set
    xs = torch.stack([train[i][0] for i in range(len(train))])
    ys = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)
    xs = xs * 2 - 1  # [0,1] -> [-1,1]
    return xs, ys, num_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--target_seed", type=int, default=0)
    ap.add_argument("--pkeep", type=float, default=0.5)
    ap.add_argument("--n_records", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None,
                    help="Exclusive upper bound on member index (for sharding across "
                         "parallel workers into disjoint ranges / separate dirs).")
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()

    device = "cuda"
    root = _HERE  # CIFAR auto-downloads to ./data/cifar{10,100} on first use

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.dataset}...")
    xs, ys, num_classes = load_cifar(args.dataset, root)
    members = compute_member_indices(len(xs), args.target_seed, args.pkeep)["keep_full_idx"]
    recon_idx = members[:args.n_records]
    print(f"  {len(recon_idx)} target members; reconstructing in batches of {args.batch_size}")

    model = build_model(args.target_seed, num_classes, device)
    params, buffers, batched_grad = make_grad_fns(model)

    meta_path = out / "meta.json"
    records = []
    if meta_path.exists():
        records = json.loads(meta_path.read_text()).get("records", [])
    done = {int(r["i"]) for r in records}

    end = args.end if args.end is not None else len(recon_idx)
    t0 = time.time()
    for b0 in range(args.start, end, args.batch_size):
        b1 = min(b0 + args.batch_size, end)
        todo = [i for i in range(b0, b1) if i not in done]
        if not todo:
            continue
        rids = recon_idx[todo]
        xt = xs[rids].to(device); yt = ys[rids].to(device)
        rec = invert_batch_robust(model, params, buffers, batched_grad, xt, yt,
                                  iters=args.iters, lr=args.lr, tv_weight=args.tv_weight, device=device)
        for j, i in enumerate(todo):
            rid = int(recon_idx[i])
            torch.save(rec[j:j + 1].cpu(), out / f"rec_{i:04d}.pt")
            save_image((rec[j:j + 1].cpu() + 1) / 2, out / f"rec_{i:04d}.png")
            mse = (rec[j].cpu() - xs[rid]).pow(2).mean().item()
            records.append({"i": i, "rec_id": rid, "y": int(ys[rid]), "mse": mse})
        records.sort(key=lambda r: r["i"])
        meta_path.write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
        per = (time.time() - t0) / (b1 - args.start)
        eta = (len(recon_idx) - b1) * per / 60
        mses = [r["mse"] for r in records[-len(todo):]]
        print(f"  batch {b0}..{b1}: mean_mse={np.mean(mses):.4f} "
              f"({per:.1f}s/img, ETA {eta:.1f} min)")

    print(f"done. total {time.time() - t0:.1f}s for {len(recon_idx)} records.")


if __name__ == "__main__":
    main()
