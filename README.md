# proust-inference

Inference code for [proust](https://huggingface.co/nappenstance/proust_v0) protein language models.

## Setup

Tested on Ubuntu 24.04 with CUDA 13.0 (B200 and B300 GPUs, sm100). Requires Python 3.12.

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone and create venv
git clone https://github.com/Furkan9015/proust-inference.git
cd proust-inference
uv venv
source .venv/bin/activate

# 3. Install PyTorch nightly (pinned version tested to work)
uv pip install torch==2.11.0.dev20260202 --index-url https://download.pytorch.org/whl/nightly/cu130

# 4. Install proust-inference with FA4 dependencies
uv pip install -e ".[fa4]"

# 5. Build FlashAttention 4 (cutedsl backend, not the usual eternal FA compilation!)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/flash_attn/cute
uv pip install -e . --no-build-isolation
cd ../../..

# 6. Install causal-conv1d (Canon layer convolutions)
git clone https://github.com/Furkan9015/causal-conv1d.git
cd causal-conv1d && uv pip install -e . --no-build-isolation && cd ..
```

## Usage

### Load model

```python
from proust_inference import load_model

# Downloads checkpoint from HuggingFace on first call, loads to cuda in bfloat16
model = load_model()
```

Or from a local checkpoint:
```python
model = load_model(local_path="/path/to/checkpoint.pt")
```

### Score a protein sequence (log-likelihood)

```python
import torch
from proust_inference import load_model, tokenize

model = load_model()

ids = tokenize("MKTLLILAVLCLGFASSALA", device="cuda")
with torch.no_grad():
    logits = model(ids.unsqueeze(0))  # (1, seq_len, vocab_size)

# Per-token log probabilities
log_probs = logits.float().log_softmax(dim=-1)
# Shift: predict token t+1 from position t
token_log_probs = log_probs[0, :-1].gather(1, ids[1:].unsqueeze(1)).squeeze(1)
print(f"Mean log-likelihood: {token_log_probs.mean().item():.4f}")
```

### Extract embeddings

```python
import torch
from proust_inference import load_model, tokenize

model = load_model()

ids = tokenize("MKTLLILAVLCLGFASSALA", device="cuda")
with torch.no_grad():
    hidden = model.get_embeddings(ids.unsqueeze(0))  # (1, seq_len, 1024)

# Mean pooling (excluding <cls> and <eos>)
embedding = hidden[0, 1:-1].mean(dim=0)  # (1024,)
```

### Batch scoring

```python
import torch
from proust_inference import load_model
from proust_inference.tokenizer import tokenize_batch

model = load_model()

sequences = ["MKTLLILAVL", "ACDEGFHIKL", "MNPQRSTVWY"]
ids, lengths = tokenize_batch(sequences, device="cuda")

with torch.no_grad():
    logits = model(ids)  # (batch, max_len, vocab_size)
```

## Reproducing ProteinGym Indels Results

### 1. Clone ProteinGym and download DMS data

```bash
git clone https://github.com/OATML-Markslab/ProteinGym.git
cd ProteinGym
curl -O https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_indels.zip
unzip DMS_ProteinGym_indels.zip && rm DMS_ProteinGym_indels.zip
cd ..
```

### 2. Run scoring

```bash
python score_proteingym.py \
    --dms-dir ProteinGym/DMS_ProteinGym_indels \
    --reference-file ProteinGym/reference_files/DMS_indels.csv \
    --output-dir scores/indels \
    --batch-size 128 \
    --indel-mode
```

The checkpoint is downloaded automatically from HuggingFace on first run. To use a local checkpoint instead, pass `--checkpoint /path/to/checkpoint.pt`.

This scores all indel DMS assays by computing per-variant log-likelihoods with `--indel-mode`, which handles insertions and deletions against the wild-type sequence. Results (per-assay CSVs with Spearman correlations) are written to `--output-dir`.

## Model

The default checkpoint (`nappenstance/proust_v0`) is a 309M parameter GQA-S2 Transformer trained on protein sequences.

- **Architecture**: GQA-S2 (Grouped Query Attention with S2 KV-sharing and VO-RoPE)
- **Hidden dim**: 1024, 24 layers, 16 heads, 2 KV heads
- **Head dim**: 128 (96 NoPE + 32 RoPE)
- **Canon ACD layers**, key offset, 5 value embeddings, optional sigmoid softcap
- **Vocab**: 32 tokens (ESM-style: 20 standard amino acids + special tokens + padding)
- **License**: CC-BY-NC-SA-2.0
