# Checkpoint Conversion

Convert MaxText Orbax/OCDBT training checkpoints into two formats:

| Output | Location | Use for |
|---|---|---|
| **HuggingFace safetensors** | `{lustre_base}/hf_converted/step_{N}/` | inference, evaluation, HF Hub upload |
| **Params-only Orbax** | `{lustre_base}/param_only/step_{N}/` | mid-training restarts (no optimizer state) |

Supports **Qwen3 dense** (`qwen3-0.6b`, `qwen3-1.7b`, …) and **Qwen3 MoE** (`qwen3-30b-a3b`, `qwen3-235b-a22b`, …) model families, including custom-scaled variants.

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd maxtext-ziq
```

### 2. Install dependencies

```bash
# For TPU post-training (includes JAX, Orbax, etils)
pip install -e ".[tpu-post-train]"

# Or minimal install for conversion only
pip install jax orbax-checkpoint etils transformers pyyaml
```

### 3. Set your HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
```

The token is needed to fetch tokenizer configs during HF conversion.
Get yours from: https://huggingface.co/settings/tokens

---

## Quick Start

### Step 1 — Set your paths in the config

Open the config file for your model type and update the two path fields:

**For MoE models** — open `configs/qwen3-30b-a3b-moe.yaml`:

```yaml
lustre_base: /lustre-data/my-model-run        # ← change to your Lustre directory
checkpoint_subdir: my-run-name/checkpoints    # ← change to your training run name
```

**For dense models** — open `configs/qwen3-0.6b-dense.yaml`:

```yaml
lustre_base: /lustre-data/my-dense-run        # ← change to your Lustre directory
checkpoint_subdir: my-run-name/checkpoints    # ← change to your training run name
```

`lustre_base` is the root folder on Lustre where your model lives.  
`checkpoint_subdir` is the path inside that folder where the training saved checkpoints.

The checkpoints are expected at:
```
{lustre_base}/{checkpoint_subdir}/{step}/items
```

For example, if your setup looks like this:
```
/lustre-data/qwen3-4b-moe/
  my-training-run/
    checkpoints/
      195000/
        items/       ← this is what the converter reads
```
Then set:
```yaml
lustre_base: /lustre-data/qwen3-4b-moe
checkpoint_subdir: my-training-run/checkpoints
```

**If you are using a MoE model with custom dimensions** (scaled up or down from the
base architecture), also update the architecture fields at the bottom of the config
to exactly match what you used in your training command:

```yaml
base_emb_dim: 2560          # ← must match training
base_num_query_heads: 20    # ← must match training
# ... etc
```

### Step 2 — Run

```bash
# Convert one step — creates both HF safetensors and params-only outputs
python src/maxtext/checkpoint_conversion/convert.py \
    --config src/maxtext/checkpoint_conversion/configs/qwen3-30b-a3b-moe.yaml \
    --step 195000

# Scan all checkpoints and convert every step not yet converted
python src/maxtext/checkpoint_conversion/convert.py \
    --config src/maxtext/checkpoint_conversion/configs/qwen3-30b-a3b-moe.yaml \
    --all
```

---

## Output Structure

```
{lustre_base}/
  hf_converted/
    step_195000/
      config.json
      model-00001-of-00004.safetensors
      model-00002-of-00004.safetensors
      model-00003-of-00004.safetensors
      model-00004-of-00004.safetensors
      model.safetensors.index.json
      tokenizer.json
      tokenizer_config.json
      chat_template.jinja
  param_only/
    step_195000/
      0/
        items/                  ← Orbax OCDBT, bfloat16, params only
  conversion_logs/
    step_195000_hf.log
    step_195000_params.log
```

---

## All CLI Options

```
python convert.py --config <yaml> --step <N>                   convert one step (both outputs)
python convert.py --config <yaml> --all                        convert all unprocessed steps
python convert.py --config <yaml> --step <N> --hf-only         HF safetensors only
python convert.py --config <yaml> --step <N> --params-only     params-only Orbax only
```

---

## Config Reference

### Required fields

| Field | Description |
|---|---|
| `model_type` | `moe` or `dense` |
| `maxtext_model_name` | model name from `utils/hf_model_configs.py` |
| `lustre_base` | root directory of this model on Lustre |
| `checkpoint_subdir` | path to checkpoints relative to `lustre_base` |

### Optional fields

| Field | Default | Description |
|---|---|---|
| `hf_out_subdir` | `hf_converted` | subfolder name for HF output |
| `params_out_subdir` | `param_only` | subfolder name for params-only output |
| `weight_dtype` | `bfloat16` | output weight dtype |
| `override_model_architecture` | `false` | set `true` for custom-scaled architectures |

### Architecture override fields (MoE, when `override_model_architecture: true`)

These must **exactly match** the values used during training:

| Field | Description |
|---|---|
| `base_emb_dim` | hidden size / embedding dimension |
| `base_num_query_heads` | number of query attention heads |
| `base_num_kv_heads` | number of KV heads (GQA) |
| `base_num_decoder_layers` | number of transformer layers |
| `head_dim` | per-head dimension |
| `base_mlp_dim` | dense MLP intermediate dimension |
| `base_moe_mlp_dim` | MoE expert intermediate dimension |
| `num_experts` | total number of experts |
| `num_experts_per_tok` | experts activated per token |
| `vocab_size` | vocabulary size |

Valid model names are listed in:
`src/maxtext/checkpoint_conversion/utils/hf_model_configs.py`

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Config file not found` | wrong path | run from repo root; use path relative to repo root |
| `lustre_base not found` | Lustre not mounted | check `df -h`; verify mount |
| `Checkpoint not found` | wrong step or subdir | the script lists available steps |
| `Insufficient disk space` | < 20 GB free | free up space; each step needs ~10 GB HF + ~8 GB params |
| `HF_TOKEN not set` | missing env var | `export HF_TOKEN=hf_your_token` |
| `Unknown maxtext_model_name` | typo in config | check `utils/hf_model_configs.py` for valid names |
| `Missing required fields` | incomplete config | add the listed fields; see example configs |
| `Missing package: jax / orbax` | deps not installed | `pip install -e ".[tpu-post-train]"` |

---

## Other conversion scripts

For converting **HuggingFace → MaxText** (opposite direction, e.g. to start training
from a public HF checkpoint), see `standalone_scripts/`:

| Script | Model family |
|---|---|
| `convert_qwen3_moe.py` | Qwen3 MoE |
| `convert_qwen3_next_scanned.py` | Qwen3-Next |
| `convert_gemma3_chkpt.py` | Gemma 3 |
| `convert_deepseek_family_ckpt.py` | DeepSeek |
