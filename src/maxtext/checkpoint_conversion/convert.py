#!/usr/bin/env python3
# Copyright 2025 ZenteiQ AiTech Innovations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""
ZIQ Checkpoint Converter

Converts a MaxText Orbax/OCDBT training checkpoint into:
  1. HuggingFace safetensors  →  {lustre_base}/hf_converted/step_{N}/
  2. Params-only Orbax        →  {lustre_base}/param_only/step_{N}/

Supports Qwen3 dense and Qwen3 MoE model families.

Usage:
    python convert.py --config configs/qwen3-30b-a3b-moe.yaml --step 195000
    python convert.py --config configs/qwen3-0.6b-dense.yaml  --all
    python convert.py --config configs/my-model.yaml          --step 195000 --hf-only
    python convert.py --config configs/my-model.yaml          --step 195000 --params-only
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ── Repo layout (auto-detected, never changes) ────────────────────────────────
# This file lives at:  src/maxtext/checkpoint_conversion/convert.py
#                      [0]  [1]     [2]                  [3 = this file]
# So REPO_ROOT is 3 levels up.
REPO_ROOT       = Path(__file__).resolve().parents[3]
TO_HF_SCRIPT    = REPO_ROOT / "src/maxtext/checkpoint_conversion/to_huggingface.py"
BASE_YML        = REPO_ROOT / "src/maxtext/configs/base.yml"
HF_CONFIGS_FILE = REPO_ROOT / "src/maxtext/checkpoint_conversion/utils/hf_model_configs.py"

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_DISK_GB = 20

# Architecture params passed to to_huggingface.py when override=True
ARCH_KEYS = [
    "base_emb_dim",
    "base_num_query_heads",
    "base_num_kv_heads",
    "base_num_decoder_layers",
    "head_dim",
    "base_mlp_dim",
    "base_moe_mlp_dim",
    "num_experts",
    "num_experts_per_tok",
    "vocab_size",
    "norm_topk_prob",
    "use_qk_norm",
    "rope_max_timescale",
    "shared_experts",
    "normalization_layer_epsilon",
]

REQUIRED_ARCH_KEYS_MOE = [
    "base_emb_dim",
    "base_num_query_heads",
    "base_num_kv_heads",
    "base_num_decoder_layers",
    "head_dim",
    "base_moe_mlp_dim",
    "num_experts",
    "num_experts_per_tok",
    "vocab_size",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hr(width=65):
    return "─" * width


def _registered_model_names() -> list[str]:
    """Read model names registered in hf_model_configs.py without importing it."""
    names = []
    try:
        with open(HF_CONFIGS_FILE) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('"') and '":' in stripped:
                    name = stripped.split('"')[1]
                    if name:
                        names.append(name)
    except Exception:
        pass
    return names or ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b",
                     "qwen3-14b", "qwen3-32b", "qwen3-30b-a3b",
                     "qwen3-30b-a3b-base", "qwen3-235b-a22b"]


# ── Config loading and validation ─────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"\nERROR: Config file not found: {config_path}")
        print(f"")
        print(f"  Make sure you are running from the repo root and the path is correct.")
        print(f"  Example:")
        print(f"    python src/maxtext/checkpoint_conversion/convert.py \\")
        print(f"        --config src/maxtext/checkpoint_conversion/configs/qwen3-30b-a3b-moe.yaml \\")
        print(f"        --step 195000")
        print(f"")
        print(f"  Available example configs:")
        configs_dir = Path(__file__).parent / "configs"
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.yaml")):
                print(f"    {f.relative_to(REPO_ROOT)}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or not cfg:
        print(f"\nERROR: Config file is empty or not valid YAML: {config_path}")
        print(f"  Copy one of the example configs and fill in your values.")
        sys.exit(1)

    return cfg


def validate_config(cfg: dict, config_path: Path):
    errors = []

    # Required top-level keys
    for key in ("model_type", "maxtext_model_name", "lustre_base", "checkpoint_subdir"):
        if key not in cfg:
            errors.append(key)

    if errors:
        print(f"\nERROR: Missing required fields in {config_path}:")
        for k in errors:
            print(f"  - {k}")
        print(f"")
        print(f"  Required fields:")
        print(f"    model_type          : moe  or  dense")
        print(f"    maxtext_model_name  : e.g. qwen3-30b-a3b  or  qwen3-0.6b")
        print(f"    lustre_base         : /lustre-data/my-model-run")
        print(f"    checkpoint_subdir   : my-run-name/checkpoints")
        sys.exit(1)

    # model_type
    model_type = str(cfg.get("model_type", "")).lower()
    if model_type not in ("moe", "dense"):
        print(f"\nERROR: Invalid model_type '{cfg.get('model_type')}' in {config_path}")
        print(f"")
        print(f"  Set model_type to one of:")
        print(f"    moe    — for Qwen3 MoE models  (qwen3-30b-a3b, qwen3-235b-a22b, ...)")
        print(f"    dense  — for Qwen3 dense models (qwen3-0.6b, qwen3-1.7b, qwen3-8b, ...)")
        sys.exit(1)

    # maxtext_model_name
    valid_names = _registered_model_names()
    model_name = cfg.get("maxtext_model_name", "")
    if model_name not in valid_names:
        qwen_names = [n for n in valid_names if "qwen" in n.lower()]
        print(f"\nERROR: Unknown maxtext_model_name '{model_name}' in {config_path}")
        print(f"")
        print(f"  Valid Qwen model names (from hf_model_configs.py):")
        for n in qwen_names:
            print(f"    {n}")
        print(f"")
        print(f"  Full list: {HF_CONFIGS_FILE.relative_to(REPO_ROOT)}")
        sys.exit(1)

    # lustre_base
    lustre_base = Path(cfg["lustre_base"])
    if not lustre_base.exists():
        print(f"\nERROR: lustre_base not found: {lustre_base}")
        print(f"")
        print(f"  Check that the Lustre filesystem is mounted:")
        print(f"    df -h {lustre_base.parent}")
        print(f"")
        print(f"  If the path is correct but does not exist yet, create it:")
        print(f"    mkdir -p {lustre_base}")
        sys.exit(1)

    # checkpoint_subdir
    ckpt_dir = lustre_base / cfg["checkpoint_subdir"]
    if not ckpt_dir.exists():
        print(f"\nERROR: Checkpoint directory not found: {ckpt_dir}")
        print(f"")
        print(f"  The converter expects checkpoints at:")
        print(f"    {{lustre_base}}/{{checkpoint_subdir}}/{{step}}/items")
        print(f"  e.g.: {ckpt_dir}/195000/items")
        print(f"")
        print(f"  Check the 'checkpoint_subdir' value in your config.")
        sys.exit(1)

    # MoE architecture fields when override=True
    if model_type == "moe" and cfg.get("override_model_architecture", True):
        missing = [k for k in REQUIRED_ARCH_KEYS_MOE if k not in cfg]
        if missing:
            print(f"\nERROR: MoE config requires architecture fields when override_model_architecture: true")
            print(f"  Missing from {config_path}:")
            for k in missing:
                print(f"    - {k}")
            print(f"")
            print(f"  These must exactly match the values used during training.")
            print(f"  See configs/qwen3-30b-a3b-moe.yaml for a reference.")
            sys.exit(1)


def _available_steps(cfg: dict) -> list[int]:
    """Return sorted list of step numbers that have a valid items/ directory."""
    ckpt_dir = Path(cfg["lustre_base"]) / cfg["checkpoint_subdir"]
    steps = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "items").exists():
            steps.append(int(p.name))
    return sorted(steps)


def _check_disk(path: Path, required_gb: float = MIN_DISK_GB) -> tuple[bool, float]:
    avail = shutil.disk_usage(str(path)).free / (1024 ** 3)
    return avail >= required_gb, avail


# ── HF conversion ─────────────────────────────────────────────────────────────

def _build_hf_command(cfg: dict, step: int, items_path: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(TO_HF_SCRIPT),
        str(BASE_YML),
        f"model_name={cfg['maxtext_model_name']}",
        f"load_parameters_path={items_path}",
        f"base_output_directory={out_dir}",
        f"weight_dtype={cfg.get('weight_dtype', 'bfloat16')}",
        "skip_jax_distributed_system=True",
        f"run_name={cfg['maxtext_model_name']}-step-{step}",
        "checkpoint_storage_use_ocdbt=True",
        "checkpoint_storage_use_zarr3=True",
    ]
    if cfg.get("override_model_architecture", False):
        cmd.append("--override_model_architecture")
        for key in ARCH_KEYS:
            if key in cfg:
                cmd.append(f"{key}={cfg[key]}")
    return cmd


def run_hf_conversion(cfg: dict, step: int, items_path: Path, out_dir: Path, log_file: Path) -> bool:
    index_file = out_dir / "model.safetensors.index.json"
    if index_file.exists():
        print(f"    [hf]     already done — skipping")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_hf_command(cfg, step, items_path, out_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env["JAX_PLATFORMS"] = "cpu"

    if not env.get("HF_TOKEN"):
        print(f"\nERROR: HF_TOKEN environment variable is not set.")
        print(f"")
        print(f"  The conversion needs a HuggingFace token to download tokenizer configs.")
        print(f"  Get your token from https://huggingface.co/settings/tokens then run:")
        print(f"    export HF_TOKEN=hf_your_token_here")
        return False

    print(f"    [hf]     converting ...  log → {log_file.name}")
    t0 = time.time()

    with open(log_file, "w") as lf:
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    [hf]     FAILED ({elapsed:.0f}s) — {log_file}")
        with open(log_file) as f:
            lines = f.readlines()
        print(f"    Last output:")
        for line in lines[-8:]:
            print(f"      {line.rstrip()}")
        return False

    if not index_file.exists():
        print(f"    [hf]     FAILED — index file missing after conversion. Check {log_file}")
        return False

    size_gb = sum(f.stat().st_size for f in out_dir.rglob("*.safetensors")) / (1024 ** 3)
    print(f"    [hf]     done  ({elapsed:.0f}s, {size_gb:.1f} GB)  →  {out_dir}")
    return True


# ── Params-only extraction ────────────────────────────────────────────────────

def run_params_extraction(cfg: dict, step: int, items_path: Path, out_dir: Path, log_file: Path) -> bool:
    out_items = out_dir / "0" / "items"
    if out_items.exists() and any(out_items.iterdir()):
        print(f"    [params] already done — skipping")
        return True

    print(f"    [params] extracting ...  log → {log_file.name}")
    t0 = time.time()

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import orbax.checkpoint as ocp
        from etils import epath
    except ImportError as e:
        print(f"    [params] FAILED — missing package: {e}")
        print(f"")
        print(f"    Install dependencies and retry:")
        print(f"      pip install -e '.[tpu-post-train]'")
        print(f"    Or minimal install:")
        print(f"      pip install jax orbax-checkpoint etils")
        return False

    try:
        jax.config.update("jax_platform_name", "cpu")

        ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(
            use_ocdbt=True, use_zarr3=True, restore_concurrent_gb=96,
        ))
        meta = ckptr.metadata(epath.Path(str(items_path)))

        devices  = np.array(jax.devices()).reshape((-1,))
        mesh     = jax.sharding.Mesh(devices, ("x",))
        no_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        restore_args = jax.tree_util.tree_map(
            lambda x: ocp.ArrayRestoreArgs(sharding=no_shard) if hasattr(x, "shape") else None,
            meta.item_metadata.tree,
            is_leaf=lambda x: hasattr(x, "shape"),
        )

        full_ckpt   = ckptr.restore(epath.Path(str(items_path)), restore_args=restore_args)
        params      = full_ckpt["params"]
        params_bf16 = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "astype") else x,
            params,
        )

        save_path = epath.Path(str(out_dir / "0"))
        save_path.mkdir(parents=True, exist_ok=True)

        save_ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(
            use_ocdbt=True, use_zarr3=True, save_concurrent_gb=96,
        ))
        save_ckptr.save(save_path / "items", params_bf16)

        elapsed  = time.time() - t0
        size_gb  = shutil.disk_usage(str(out_dir)).used / (1024 ** 3)
        print(f"    [params] done  ({elapsed:.0f}s, {size_gb:.1f} GB)  →  {out_dir}")

        with open(log_file, "w") as lf:
            lf.write(f"step:    {step}\n")
            lf.write(f"source:  {items_path}\n")
            lf.write(f"output:  {out_dir}\n")
            lf.write(f"elapsed: {elapsed:.1f}s\n")
            lf.write(f"size:    {size_gb:.1f} GB\n")
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        import traceback
        with open(log_file, "w") as lf:
            traceback.print_exc(file=lf)
        print(f"    [params] FAILED ({elapsed:.0f}s): {exc}")
        print(f"    Full traceback: {log_file}")
        return False


# ── Per-step orchestration ────────────────────────────────────────────────────

def convert_step(cfg: dict, step: int, hf_only: bool, params_only_flag: bool) -> dict:
    lustre_base = Path(cfg["lustre_base"])
    items_path  = lustre_base / cfg["checkpoint_subdir"] / str(step) / "items"
    hf_dir      = lustre_base / cfg.get("hf_out_subdir",     "hf_converted") / f"step_{step}"
    params_dir  = lustre_base / cfg.get("params_out_subdir", "param_only")   / f"step_{step}"
    log_dir     = lustre_base / "conversion_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'─'*55}")
    print(f"  Step {step:,}")
    print(f"    source   {items_path}")
    print(f"    hf out   {hf_dir}")
    print(f"    params   {params_dir}")

    # Source check
    if not items_path.exists():
        available = _available_steps(cfg)
        print(f"\n    ERROR: Checkpoint not found: {items_path}")
        if available:
            print(f"    Available steps in this run: {', '.join(str(s) for s in available)}")
        else:
            print(f"    No valid checkpoints found in: {lustre_base / cfg['checkpoint_subdir']}")
            print(f"    Each step folder must contain an 'items' subdirectory.")
        return {"step": step, "hf": False, "params": False}

    # Disk space check
    ok, avail_gb = _check_disk(lustre_base)
    if not ok:
        print(f"\n    ERROR: Not enough disk space.")
        print(f"    Available: {avail_gb:.1f} GB   Required: {MIN_DISK_GB} GB")
        print(f"    Free up space on {lustre_base} and retry.")
        return {"step": step, "hf": False, "params": False}

    hf_ok     = True
    params_ok = True

    if not params_only_flag:
        hf_ok = run_hf_conversion(
            cfg, step, items_path, hf_dir,
            log_dir / f"step_{step}_hf.log",
        )

    if not hf_only:
        params_ok = run_params_extraction(
            cfg, step, items_path, params_dir,
            log_dir / f"step_{step}_params.log",
        )

    return {"step": step, "hf": hf_ok, "params": params_ok}


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], cfg: dict, hf_only: bool, params_only_flag: bool):
    lustre_base = Path(cfg["lustre_base"])

    print(f"\n{'═'*65}")
    print(f"  SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Step':<14}  {'HF':^14}  {'Params':^14}")
    print(f"  {'─'*14}  {'─'*14}  {'─'*14}")

    for r in results:
        hf_str = (
            "✓  done"  if r["hf"]
            else "—  skipped" if params_only_flag
            else "✗  failed"
        )
        params_str = (
            "✓  done"  if r["params"]
            else "—  skipped" if hf_only
            else "✗  failed"
        )
        print(f"  {r['step']:<14,}  {hf_str:^14}  {params_str:^14}")

    print(f"{'═'*65}")
    print(f"\n  Output root : {lustre_base}")
    print(f"  HF files    : {cfg.get('hf_out_subdir', 'hf_converted')}/step_{{N}}/")
    print(f"  Params-only : {cfg.get('params_out_subdir', 'param_only')}/step_{{N}}/0/items")
    print(f"  Logs        : conversion_logs/step_{{N}}_hf.log  |  step_{{N}}_params.log")

    failed = [r["step"] for r in results if not r["hf"] or not r["params"]]
    if failed:
        print(f"\n  Failed steps: {', '.join(str(s) for s in failed)}")
        print(f"  Check the logs above for details.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="convert.py",
        description="Convert MaxText Orbax checkpoints to HuggingFace and params-only formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Convert step 195000 (both HF and params-only)
  python convert.py --config configs/qwen3-30b-a3b-moe.yaml --step 195000

  # Convert all unprocessed steps automatically
  python convert.py --config configs/qwen3-30b-a3b-moe.yaml --all

  # HF safetensors only
  python convert.py --config configs/qwen3-30b-a3b-moe.yaml --step 195000 --hf-only

  # Params-only Orbax only
  python convert.py --config configs/qwen3-30b-a3b-moe.yaml --step 195000 --params-only
""",
    )
    parser.add_argument("--config",        required=True, type=Path,
                        help="path to model YAML config  (e.g. configs/qwen3-30b-a3b-moe.yaml)")
    parser.add_argument("--step",          type=int, default=None,
                        help="checkpoint step number to convert")
    parser.add_argument("--all",           action="store_true",
                        help="convert every unprocessed step found in checkpoint_subdir")
    parser.add_argument("--hf-only",      action="store_true",
                        help="run HF conversion only, skip params-only extraction")
    parser.add_argument("--params-only",  action="store_true",
                        help="run params-only extraction only, skip HF conversion")
    args = parser.parse_args()

    if not args.step and not args.all:
        parser.error(
            "You must specify what to convert.\n\n"
            "  Convert one step:   --step 195000\n"
            "  Convert all steps:  --all\n"
        )
    if args.step and args.all:
        parser.error("Use --step <N> OR --all, not both.")
    if args.hf_only and args.params_only:
        parser.error("--hf-only and --params-only cannot be used together.")

    cfg = load_config(args.config)
    validate_config(cfg, args.config)

    print(f"\n{'═'*65}")
    print(f"  ZIQ Checkpoint Converter")
    print(f"{'═'*65}")
    print(f"  model      : {cfg['maxtext_model_name']}  ({cfg['model_type'].upper()})")
    print(f"  lustre     : {cfg['lustre_base']}")
    print(f"  hf out     : {cfg.get('hf_out_subdir', 'hf_converted')}/step_{{N}}/")
    print(f"  params out : {cfg.get('params_out_subdir', 'param_only')}/step_{{N}}/")
    print(f"  override   : {cfg.get('override_model_architecture', False)}")

    # Determine steps to process
    if args.step:
        steps = [args.step]
    else:
        available = _available_steps(cfg)
        if not available:
            print(f"\nERROR: No checkpoints found.")
            print(f"  Scanned: {Path(cfg['lustre_base']) / cfg['checkpoint_subdir']}")
            print(f"  Expected numeric folders each containing an 'items' subdirectory.")
            sys.exit(1)

        lustre_base = Path(cfg["lustre_base"])
        hf_base     = lustre_base / cfg.get("hf_out_subdir",     "hf_converted")
        params_base = lustre_base / cfg.get("params_out_subdir", "param_only")

        pending, done = [], []
        for s in available:
            hf_done     = (hf_base / f"step_{s}" / "model.safetensors.index.json").exists()
            params_done = (params_base / f"step_{s}" / "0" / "items").exists()
            need_hf     = not hf_done     and not args.params_only
            need_params = not params_done and not args.hf_only
            if need_hf or need_params:
                pending.append(s)
            else:
                done.append(s)

        print(f"\n  Available steps  : {len(available)}  ({', '.join(str(s) for s in available)})")
        if done:
            print(f"  Already done     : {', '.join(str(s) for s in done)}")
        print(f"  To convert       : {len(pending)}  ({', '.join(str(s) for s in pending) if pending else 'none'})")

        if not pending:
            print(f"\n  Nothing to do — all steps already converted.")
            sys.exit(0)

        steps = pending

    # Run
    results = [
        convert_step(cfg, step, hf_only=args.hf_only, params_only_flag=args.params_only)
        for step in steps
    ]

    print_summary(results, cfg, hf_only=args.hf_only, params_only_flag=args.params_only)

    if any(not r["hf"] or not r["params"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
