"""GIFD-style gradient inversion attack on UTKFace / CelebA ResNet-18 targets.

Implements W+ space optimization with cosine-similarity gradient matching
(Geiping NeurIPS 2020) under the StyleGAN2-FFHQ face prior. This corresponds
to the GIFD-z baseline (PSNR 16.99 / SSIM 0.39 on FFHQ per Fang et al.
ICCV 2023, Table 8a), which is the strictly necessary mechanism to demonstrate
collusion amplification. Intermediate-feature search (Algorithm 1 lines 7-17)
is implemented as an outer loop with progressive l1-ball relaxation.

Adversary 1 (gradient inverter) sees per-sample CE gradients on a single member
record at a time and outputs the recovered image. The recovered images become
the reconstruction pool that downstream AIA/DIA attacks (Adversary 2) consume.

Usage:
    python face_recon.py --dataset utkface --target_seed 0 --y_attr sex --z_attr race \
        --method geiping --geiping_iters 6000 --n_records 100 \
        --output_dir recon_pools/utkface_seed0_geiping
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

_HERE = Path(__file__).resolve().parent
# GIFD (Fang et al., ICCV 2023) is an external dependency: clone it to ./GIFD or
# point $GIFD_ROOT at an existing checkout. See README C.2 for the install path.
_GIFD = Path(os.environ.get("GIFD_ROOT", _HERE / "GIFD"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_GIFD))
# Import GIFD's StyleGAN2 generator standalone (its genmodels dir on the path)
# rather than via `inversefed.genmodels...`: the latter runs inversefed's package
# __init__, which eagerly pulls the whole gradient-inversion library (FrEIA,
# nevergrad, BigGAN, ...). We use only the generator. The fused CUDA ops still
# JIT-compile on first import (needs ninja + a system CUDA toolchain; README C.2).
sys.path.insert(0, str(_GIFD / "inversefed" / "genmodels"))

from face_common import get_splits, load_celeba, load_utkface, to_model_input
from face_targets import build_resnet18_64, load_target

# The StyleGAN2 generator (from the GIFD checkout) is imported lazily inside main(),
# only when --method gifd is selected. The default geiping path is GAN-free, so the
# reproduction (run_all.sh, all --method geiping) needs no GIFD checkout, no
# StyleGAN2 checkpoint, and no cpp_extension/ninja toolchain.


STYLEGAN_CKPT = _GIFD / "inversefed" / "genmodels" / "stylegan2_io" / "stylegan2-ffhq-config-f.pt"


def cosine_grad_loss(dummy_grads, target_grads):
    cost, pn0, pn1 = 0.0, 0.0, 0.0
    for dg, tg in zip(dummy_grads, target_grads):
        cost = cost - (dg * tg).sum()
        pn0 = pn0 + dg.pow(2).sum()
        pn1 = pn1 + tg.pow(2).sum()
    return 1 + cost / (pn0.sqrt() * pn1.sqrt() + 1e-12)


def tv_loss(img):
    return (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean() + \
           (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()


def project_l1(x, ref, radius):
    delta = x - ref
    flat = delta.reshape(delta.size(0), -1)
    l1 = flat.abs().sum(dim=1, keepdim=True)
    scale = torch.where(l1 > radius, radius / (l1 + 1e-12), torch.ones_like(l1))
    return ref + (flat * scale).reshape_as(delta)


def init_w_plus(generator, latent_avg, device, truncation=0.7, jitter=0.0):
    n_latent = generator.n_latent
    base = latent_avg.view(1, 1, -1).repeat(1, n_latent, 1)
    if jitter > 0:
        # Sample z, run mapping, blend toward latent_avg by truncation.
        z = torch.randn(1, generator.style_dim, device=device)
        w = generator.style(z).unsqueeze(1).repeat(1, n_latent, 1)
        return base + truncation * (w - base) + jitter * torch.randn_like(base)
    return base + truncation * 0.1 * torch.randn_like(base)


def generate_64(generator, w_plus):
    img, _ = generator([w_plus], input_is_latent=True, randomize_noise=False)
    return F.interpolate(img, size=64, mode="bilinear", align_corners=False)


def run_phase(target_model, target_grads, label_t, generator,
              w_init, *, iters, lr, tv_weight, ref=None, radius=None, log_prefix=""):
    """One optimization phase. If ref/radius given, project W+ onto l1 ball around ref each step."""
    w = w_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    best = {"loss": math.inf, "img": None, "w": None}
    log = []
    for step in range(iters):
        opt.zero_grad()
        img = generate_64(generator, w)
        logits = target_model(img)
        ce = F.cross_entropy(logits, label_t)
        dummy_grads = torch.autograd.grad(ce, target_model.parameters(), create_graph=True)
        loss = cosine_grad_loss(dummy_grads, target_grads) + tv_weight * tv_loss(img)
        loss.backward()
        opt.step()
        sched.step()
        if ref is not None and radius is not None:
            with torch.no_grad():
                w.data = project_l1(w.data, ref, radius)
        if loss.item() < best["loss"]:
            best["loss"] = loss.item()
            best["img"] = img.detach().clone()
            best["w"] = w.detach().clone()
        if step % max(1, iters // 4) == 0:
            log.append((step, loss.item()))
    return best, log


def gifd_attack_single(target_model, target_grads, label, generator, latent_avg, *,
                       restarts, iters_latent, iters_feat, feat_layers,
                       radius_w_base, lr_latent, lr_feat, tv_weight, device):
    """Restart-best W+ optimization (phase 1) + progressive l1-ball feature search (phase 2)."""
    label_t = torch.tensor([label], device=device, dtype=torch.long)
    global_best = {"loss": math.inf, "img": None}

    for trial in range(restarts):
        # Phase 1: latent space search with random truncation jitter init.
        w_init = init_w_plus(generator, latent_avg, device,
                             truncation=0.7, jitter=0.03 if trial > 0 else 0.0)
        phase1, _ = run_phase(target_model, target_grads, label_t, generator, w_init,
                              iters=iters_latent, lr=lr_latent, tv_weight=tv_weight)
        if phase1["loss"] < global_best["loss"]:
            global_best.update(phase1)

        # Phase 2: progressive intermediate-feature search (l1 ball on W+).
        w_ref = phase1["w"]
        for k in range(feat_layers):
            radius = radius_w_base * (k + 1)
            phase2, _ = run_phase(target_model, target_grads, label_t, generator, w_ref,
                                  iters=iters_feat, lr=lr_feat, tv_weight=tv_weight,
                                  ref=w_ref, radius=radius)
            if phase2["loss"] < global_best["loss"]:
                global_best.update(phase2)
            w_ref = phase2["w"]

    return global_best["img"], {"best_loss": global_best["loss"]}


def compute_per_sample_grad(model, x, y):
    model.zero_grad()
    grads = torch.autograd.grad(F.cross_entropy(model(x), y), model.parameters())
    return [g.detach() for g in grads]


def geiping_attack_single(target_model, target_grads, label, *,
                          restarts, iters, lr, tv_weight, device):
    """Geiping et al. NeurIPS 2020 "Inverting Gradients": GAN-free pixel-space
    optimization with cosine-similarity gradient matching + TV regularization.

    No generative prior - the recovered image is optimised directly in pixel
    space. On a randomly-initialised model the per-sample gradient strongly
    constrains the input, so pixel-space inversion recovers the *actual* record
    rather than a prior-plausible substitute.
    """
    label_t = torch.tensor([label], device=device, dtype=torch.long)
    global_best = {"loss": math.inf, "img": None}
    for trial in range(restarts):
        x = (torch.randn(1, 3, 64, 64, device=device) * 0.1).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
        for step in range(iters):
            opt.zero_grad()
            logits = target_model(x)
            ce = F.cross_entropy(logits, label_t)
            dummy_grads = torch.autograd.grad(ce, target_model.parameters(), create_graph=True)
            loss = cosine_grad_loss(dummy_grads, target_grads) + tv_weight * tv_loss(x)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                x.clamp_(-1.0, 1.0)
            if loss.item() < global_best["loss"]:
                global_best = {"loss": loss.item(), "img": x.detach().clone()}
    return global_best["img"], {"best_loss": global_best["loss"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["utkface", "celeba"], required=True)
    ap.add_argument("--target_seed", type=int, required=True)
    ap.add_argument("--n_records", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--iters_latent", type=int, default=600)
    ap.add_argument("--iters_feat", type=int, default=200)
    ap.add_argument("--feat_layers", type=int, default=2)
    ap.add_argument("--radius_w_base", type=float, default=4.0)
    ap.add_argument("--lr_latent", type=float, default=0.03)
    ap.add_argument("--lr_feat", type=float, default=0.01)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--grad_model", choices=["init", "trained"], default="init",
                    help="Model state whose per-sample gradient is inverted. 'init' "
                         "= randomly-initialised ResNet-18 (the standard gradient-inversion "
                         "threat model: Adversary 1 observes an early FL round). 'trained' "
                         "= the converged target (gradients near-degenerate, hard to invert).")
    ap.add_argument("--method", choices=["gifd", "geiping"], default="geiping",
                    help="gifd = StyleGAN2-prior W+ search; geiping = GAN-free pixel-space "
                         "Inverting Gradients (NeurIPS 2020). geiping recovers the actual "
                         "record on init-model gradients; gifd's prior over-constrains.")
    ap.add_argument("--geiping_iters", type=int, default=4000)
    ap.add_argument("--geiping_lr", type=float, default=0.1)
    ap.add_argument("--y_attr", default=None,
                    help="UTKFace target attribute (age/race/sex); selects the gradient "
                         "label and the target checkpoint. Ignored for CelebA.")
    ap.add_argument("--z_attr", default=None, help="UTKFace sensitive attribute; ignored for CelebA.")
    args = ap.parse_args()

    device = "cuda"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # The StyleGAN2 prior is only used by --method gifd. Import and build it lazily
    # so the default geiping path (used by run_all.sh) needs no GIFD checkout, no
    # StyleGAN2 checkpoint, and no cpp_extension/ninja toolchain. geiping_attack_single
    # never references G / latent_avg, and torch.manual_seed below re-seeds the RNG,
    # so skipping this build cannot change any geiping reconstruction.
    G = None
    latent_avg = None
    if args.method == "gifd":
        from stylegan2_io.model import Generator

        G = Generator(size=1024, style_dim=512, n_mlp=8).to(device)
        ckpt = torch.load(STYLEGAN_CKPT, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["g_ema"], strict=False)
        G.eval()
        for p in G.parameters():
            p.requires_grad_(False)
        latent_avg = ckpt["latent_avg"].to(device)

    if args.dataset == "utkface":
        y_attr = args.y_attr or "race"
        z_attr = args.z_attr or ("sex" if y_attr != "sex" else "race")
        face = load_utkface(y_attr=y_attr, z_attr=z_attr)
        target_task = y_attr
    else:
        face = load_celeba()
        target_task = None
    # Splits are deterministic in (dataset size, seed), so derive them directly. The
    # recon pool is built from an init model (seed = target_seed) and is target-
    # independent, so it does not need a trained checkpoint (only --grad_model trained,
    # below, loads the target itself).
    splits = get_splits(face, args.target_seed)

    # The gradient that Adversary 1 inverts. Standard gradient-inversion attacks
    # (DLG, Geiping, GIFD) invert a randomly-initialised model's gradient - the
    # model state at the first FL round - where per-sample signal is strong. A
    # fully-converged overfit model produces near-degenerate gradients that no
    # published attack can invert; that is a property of the target, not the attack.
    if args.grad_model == "init":
        torch.manual_seed(args.target_seed)
        grad_model = build_resnet18_64().to(device)
    else:
        grad_model, _ = load_target(args.dataset, args.target_seed, task=target_task)
    grad_model.eval()
    for p in grad_model.parameters():
        p.requires_grad_(True)
    target_model = grad_model  # name used by the per-record loop below

    recon_idx = splits["recon_pool"]
    end = args.end if args.end is not None else min(args.n_records, len(recon_idx))

    # Resume-safe: if rec_*.pt files already exist, skip ahead and keep their meta.
    meta_path = out / "meta.json"
    done = sorted(int(f.stem.split("_")[1]) for f in out.glob("rec_*.pt"))
    start = args.start
    records_meta = []
    if done and meta_path.exists():
        records_meta = json.loads(meta_path.read_text()).get("records", [])
        if start == 0:
            start = max(done) + 1  # auto-resume after the last completed record
    print(f"reconstructing {start}..{end} of recon_pool (size {len(recon_idx)}); "
          f"{len(records_meta)} already done")

    t0 = time.time()
    for i in range(start, end):
        rec_id = int(recon_idx[i])
        x_target = to_model_input(face.images[rec_id:rec_id + 1]).to(device)
        y_target = torch.tensor([int(face.y[rec_id])], device=device, dtype=torch.long)
        target_grads = compute_per_sample_grad(target_model, x_target, y_target)

        if args.method == "geiping":
            recovered, info = geiping_attack_single(
                target_model=target_model, target_grads=target_grads, label=int(face.y[rec_id]),
                restarts=args.restarts, iters=args.geiping_iters, lr=args.geiping_lr,
                tv_weight=args.tv_weight, device=device,
            )
        else:
            recovered, info = gifd_attack_single(
                target_model=target_model, target_grads=target_grads, label=int(face.y[rec_id]),
                generator=G, latent_avg=latent_avg,
                restarts=args.restarts, iters_latent=args.iters_latent, iters_feat=args.iters_feat,
                feat_layers=args.feat_layers, radius_w_base=args.radius_w_base,
                lr_latent=args.lr_latent, lr_feat=args.lr_feat, tv_weight=args.tv_weight,
                device=device,
            )

        torch.save(recovered.cpu(), out / f"rec_{i:04d}.pt")
        save_image((recovered.cpu() + 1) / 2, out / f"rec_{i:04d}.png")
        with torch.no_grad():
            err = (recovered.cpu() - x_target.cpu()).pow(2).mean().item()
        records_meta.append({"i": i, "rec_id": rec_id, "y": int(face.y[rec_id]),
                             "z": int(face.z[rec_id]), "mse": err, **info})
        per = (time.time() - t0) / (i - start + 1)
        eta_min = (end - i - 1) * per / 60
        print(f"  rec {i}/{end}: mse={err:.4f}  best_loss={info['best_loss']:.3f}  "
              f"({per:.1f}s/img, ETA {eta_min:.1f} min)")
        (out / "meta.json").write_text(json.dumps({"args": vars(args), "records": records_meta}, indent=2))

    print(f"done. total {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
