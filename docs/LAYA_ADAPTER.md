# Laya Spatial Adapter — design and integration contract

## Verified upstream contract

The upstream repository `NandhaKishorM/laya` (source checked at commit
`3a38cbe83f6ef556f3125fb0065fdc5a55d05792`, 2026-11-10)
provides:

- `laya.load(model_id, device=..., revision=..., backend="eager")` loading the model checkpoint.
- `agent.model.encoder`: an encoder with `config.hidden_size` (ModernBERT-large for the English checkpoint).
- `laya.common.DecisionModel` with `encoder`, plus a **text-specific** option-marker scoring head.
- Hugging Face `ModernBertModel.forward(inputs_embeds=..., attention_mask=...)` returning `last_hidden_state`.
- Upstream PyPI version `laya==0.4.31` is the optional extra for this repository; the upstream checkpoint is `convaiinnovations/laya`, Apache-2.0.

Relevant source:
- https://github.com/NandhaKishorM/laya/blob/3a38cbe83f6ef556f3125fb0065fdc5a55d05792/laya/common.py
- https://github.com/NandhaKishorM/laya/blob/3a38cbe83f6ef556f3125fb0065fdc5a55d05792/laya/agent.py
- https://huggingface.co/docs/transformers/model_doc/modernbert
- https://huggingface.co/convaiinnovations/laya

## Explicit modeling assumptions

The pretraining task is **text classification / typed decisions**, not
partially-observable navigation. There is no theoretical reason to assume
its contextual embedding geometry transfers to spatial grids. All action
selection and the new actor/critic heads need task-specific training.
The original option-marker Decision Head is **not** reused.

Spatial pipeline:

```
belief observation (B,6,H,W), float32
      → Conv2D / GroupNorm / GELU / Downsample
      → AdaptiveAvgPool token_grid×token_grid (default 8×8)
      → Linear projection to backbone hidden size
      → Trainable 2D row-major positional embedding
      → Prepend learned summary token (length 65 at default)
      → Frozen ModernBERT via inputs_embeds, all-valid attention mask
      → Summary token hidden state
      ├── Actor head (five action logits)
      └── Critic head (one scalar, untrained by BC)
```

The pretrained encoder remains in eval mode when frozen; gradients still
flow through it into spatial token embeddings. **Do not** wrap its forward
with `no_grad`, as that would prevent training the adapter.

All train and validation data comes from the six-channel public belief
observation. Hidden simulator obstacles are NOT passed to Laya. Models
are compared at test time with the same map seeds, sensing, collision and
step-budget rules.

## Checkpoint and model provenance

`laya_cli` loads the backbone from Hugging Face using the optional
`--revision` argument. The resulting agent's resolved revision and the
installed Laya package version are recorded in the adapter checkpoint.
The adapter-only checkpoint stores **no backbone weights**. At evaluation,
the backend is loaded again at the stored resolved revision before
applying the adapter state. Only trust model artifacts and checkpoints
from known sources.

```bash
# Install Blackwell-compatible CUDA PyTorch first.
python -m pip install -e ".[train,laya]"

python -m layanav.training.laya_cli \
  --train datasets/teacher_train.npz \
  --validation datasets/teacher_val.npz \
  --output checkpoints/laya_spatial_bc.pt \
  --model-id convaiinnovations/laya \
  --revision <exact-HF-model-commit-SHA> \
  --device cuda --batch-size 2 --accumulation-steps 4 \
  --feature-channels 64 --token-grid 8 --epochs 5

# Compare with A*, CNN/BC and PPO using the same held-out seed set.
python -m layanav.evaluation.cli \
  --methods astar bc ppo laya \
  --bc-checkpoint checkpoints/cnn_bc.pt \
  --ppo-checkpoint checkpoints/ppo_standard.zip \
  --laya-checkpoint checkpoints/laya_spatial_bc.pt \
  --families sparse standard dense \
  --eval-seed 200000 --episodes 100 --sensor raycast \
  --device cuda --output-dir results/laya_comparison
```

The `<exact-HF-model-commit-SHA>` placeholder must be replaced by a
real commit SHA. A non-pinned revision is allowed but weakens
reproducibility and may change on future downloads.

## Verification boundary

CI tests a **tiny stand-in backbone**, checking the
`inputs_embeds` interface, gradients through a frozen encoder,
state-dict roundtrip, BC training and benchmark integration. It does not
download 808 MB pretrained weights, and cannot certify compatibility of
your exact Laya/Transformers/CUDA versions.

An explicit command tests the actual pretrained checkpoint on the local
CUDA GPU. First run a *kernel-level preflight* without downloading the weights:

```bash
# In the environment that runs LayaNav. Use the official CUDA-enabled
# PyTorch build appropriate for your NVIDIA driver and RTX 5070.
python -m pip install -e ".[train,laya]"

python -m layanav.training.laya_smoke \
  --device cuda --preflight-only --output results/laya_gpu_preflight.json
```

Ensure the report says `kernel_test: passed` and shows the expected GPU
and compute capability (RTX 5070 typically SM 12.0). A CUDA-visible device
does not imply a working PyTorch kernel.

Then run the **real pretrained weight** test (download about 808 MB on first use):

```bash
USE_TF=0 python -m layanav.training.laya_smoke \
  --model-id convaiinnovations/laya \
  --device cuda --movement-mode 8 --batch-size 1 \
  --token-grid 8 --feature-channels 64 --grid-size 32 \
  --seed 42 --output results/laya_gpu_smoke.json
```

This tests the frozen backbone's real `inputs_embeds` path, requires
finite actor/critic outputs, nonzero finite **tokenizer and actor**
gradients, a successful AdamW optimizer update, and valid inference
action ID. It reports resolved model revision, torch/CUDA/Transformers/Laya
versions, GPU capability, separate load/forward/backward/inference
latencies, peak device memory and trainable adapter/actor gradients.
The `--output` JSON is saved on success **and** validation failure.
Failure exits nonzero with the actual exception. The script detects and
**rejects silent upstream CPU fallback** when CUDA is requested.

If you see CUDA compatibility warnings or `no kernel image is available`,
update the CUDA-enabled PyTorch wheel from the official installer:
https://pytorch.org/get-started/locally/
For memory issues, retry with `--token-grid 4 --batch-size 1`.
Use a real Hub commit in `--revision` for reproducible follow-up
experiments. The upstream Laya project documents `USE_TF=0` to avoid
an import hang if TensorFlow is installed.

A `passed` report is an **integration test**, not evidence of successful
navigation, pretrained transfer, an accurate critic, or a trained policy.

The next required validation is a real pretrained Laya forward/backward
run on Ubuntu 24.04 / RTX 5070 (12GB), measuring peak GPU memory and
batch-size limits. If too large, lower `--batch-size` to 1, reduce
`--token-grid`, or add carefully tested activation checkpointing.
The current BC training does not train the critic and is not PPO fine-tuning.

## Research ablations

Compare, with identical spatial tokenization and training budgets:
1. Pretrained Laya frozen backbone + trainable adapter/head
2. Same-size randomly initialized backbone + adapter/head
3. Pretrained backbone partial unfreeze/LoRA (future)
4. Lightweight CNN and standard Transformer baseline
5. Online A* using identical belief state

Evidence for positive transfer requires matching architecture, optimizer,
compute/data and held-out environments. A working pretrained-weight
loader is not itself evidence of transfer or superior navigation.
