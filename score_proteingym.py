#!/usr/bin/env python3
"""Score ProteinGym DMS benchmarks with a proust model.

Computes fitness scores for protein variants by calculating
the log-likelihood of mutant sequences using the causal language model.

Usage:
    # Score a single DMS assay
    python score_proteingym.py \
        --dms-file ProteinGym/DMS_ProteinGym_substitutions/BLAT_ECOLX_Firnberg_2014.csv \
        --output scores/BLAT_ECOLX_Firnberg_2014_scores.csv

    # Score all substitution assays
    python score_proteingym.py \
        --dms-dir ProteinGym/DMS_ProteinGym_substitutions \
        --reference-file ProteinGym/reference_files/DMS_substitutions.csv \
        --output-dir scores/substitutions

    # Score all indel assays
    python score_proteingym.py \
        --dms-dir ProteinGym/DMS_ProteinGym_indels \
        --reference-file ProteinGym/reference_files/DMS_indels.csv \
        --output-dir scores/indels \
        --indel-mode
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr

from proust_inference import load_model
from proust_inference.tokenizer import ESM_TOKENS


def tokenize_sequence(seq: str) -> list[int]:
    """Tokenize a protein sequence to ESM token IDs.

    Matches training format: [AA1, AA2, ..., AAn, <eos>]
    """
    tokens = []

    for aa in seq.upper():
        if aa in ESM_TOKENS:
            tokens.append(ESM_TOKENS[aa])
        else:
            tokens.append(ESM_TOKENS['<unk>'])

    tokens.append(ESM_TOKENS['<eos>'])

    return tokens


def compute_log_likelihood(
    model,
    sequences: list[str],
    device: str = 'cuda',
    batch_size: int = 256,
) -> np.ndarray:
    """Compute log-likelihood scores for sequences using varlen attention.

    Uses causal language modeling: sum of log P(token_i | tokens_<i)
    Optimized with packed sequences (no padding waste).
    """
    model.eval()

    # Pre-tokenize all sequences at once
    print(f"  Tokenizing {len(sequences)} sequences...")
    tokenized = [tokenize_sequence(seq) for seq in sequences]

    # Sort by length for efficient batching (similar lengths together)
    indices = list(range(len(tokenized)))
    indices.sort(key=lambda i: len(tokenized[i]))
    tokenized_sorted = [tokenized[i] for i in indices]

    log_likelihoods_sorted = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_sorted), batch_size), desc="Computing scores", leave=False):
            batch_tokens = tokenized_sorted[i:i + batch_size]
            lengths = [len(t) for t in batch_tokens]

            # Pack sequences without padding
            packed = torch.cat([
                torch.tensor(t, dtype=torch.long, device=device)
                for t in batch_tokens
            ])

            # Build cu_seqlens for varlen attention
            cu_seqlens = torch.zeros(len(batch_tokens) + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = torch.cumsum(
                torch.tensor(lengths, dtype=torch.int32, device=device), dim=0
            )
            max_seqlen = max(lengths)

            # Forward pass with varlen - model returns (logits, loss)
            logits = model(packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

            # Compute log probabilities
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # Extract per-sequence scores
            for j, length in enumerate(lengths):
                start = cu_seqlens[j].item()
                end = cu_seqlens[j + 1].item()

                # Shift for causal LM: predict token[t+1] from logits[t]
                seq_log_probs = log_probs[start:end-1]  # [seq_len-1, vocab]
                seq_targets = packed[start+1:end]        # [seq_len-1]

                # Gather target log probs
                target_log_probs = seq_log_probs.gather(1, seq_targets.unsqueeze(-1)).squeeze(-1)

                # Normalize by number of predictions (length - 1)
                seq_score = target_log_probs.sum() / (length - 1)
                log_likelihoods_sorted.append(seq_score.item())

    # Unsort to original order
    log_likelihoods = [0.0] * len(sequences)
    for sorted_idx, orig_idx in enumerate(indices):
        log_likelihoods[orig_idx] = log_likelihoods_sorted[sorted_idx]

    return np.array(log_likelihoods)


def compute_masked_marginal_score(
    model,
    wt_sequence: str,
    mutant_sequences: list[str],
    mutant_positions: list[list[int]],
    device: str = 'cuda',
    batch_size: int = 8,
) -> np.ndarray:
    """Compute masked marginal scores for mutations.

    For each mutation at position i:
    score = log P(mut_aa | context) - log P(wt_aa | context)

    This is typically more accurate than full sequence log-likelihood
    for single and few-site mutations.
    """
    model.eval()
    scores = []

    # Tokenize wildtype
    wt_tokens = tokenize_sequence(wt_sequence)

    with torch.no_grad():
        # Get wildtype predictions using varlen (single sequence, no padding)
        wt_input = torch.tensor(wt_tokens, dtype=torch.long, device=device)
        cu_seqlens = torch.tensor([0, len(wt_tokens)], dtype=torch.int32, device=device)
        wt_logits = model(wt_input, cu_seqlens=cu_seqlens, max_seqlen=len(wt_tokens))
        wt_log_probs = F.log_softmax(wt_logits.float(), dim=-1)  # [seq_len, vocab]

        for i in tqdm(range(0, len(mutant_sequences), batch_size), desc="Computing marginal scores", leave=False):
            batch_seqs = mutant_sequences[i:i + batch_size]
            batch_positions = mutant_positions[i:i + batch_size]

            for seq, positions in zip(batch_seqs, batch_positions):
                mut_tokens = tokenize_sequence(seq)

                score = 0.0
                for pos in positions:
                    # Position in tokenized sequence (direct, no <cls>)
                    tok_pos = pos
                    if tok_pos >= len(wt_tokens) - 1 or tok_pos >= len(mut_tokens) - 1:
                        continue

                    wt_token = wt_tokens[tok_pos]
                    mut_token = mut_tokens[tok_pos]

                    # Use position tok_pos - 1 to predict tok_pos (causal LM)
                    if tok_pos > 0:
                        log_p_mut = wt_log_probs[tok_pos - 1, mut_token].item()
                        log_p_wt = wt_log_probs[tok_pos - 1, wt_token].item()
                        score += log_p_mut - log_p_wt

                scores.append(score)

    return np.array(scores)


def parse_mutant_string(mutant: str, start_idx: int = 1) -> list[int]:
    """Parse mutation string like 'A123G' or 'A123G:B456C' to get positions."""
    positions = []
    for mut in mutant.split(':'):
        try:
            pos = int(mut[1:-1]) - start_idx
            positions.append(pos)
        except (ValueError, IndexError):
            continue
    return positions


def score_dms_file(
    model,
    dms_file: str,
    target_seq: str = None,
    device: str = 'cuda',
    batch_size: int = 8,
    scoring_method: str = 'log_likelihood',
    indel_mode: bool = False,
) -> pd.DataFrame:
    """Score a single DMS file."""
    df = pd.read_csv(dms_file, low_memory=False)

    if 'mutated_sequence' not in df.columns:
        raise ValueError(f"DMS file must have 'mutated_sequence' column: {dms_file}")

    sequences = df['mutated_sequence'].tolist()

    if scoring_method == 'log_likelihood':
        scores = compute_log_likelihood(
            model, sequences, device, batch_size
        )
    elif scoring_method == 'masked_marginal' and not indel_mode and target_seq:
        # Parse mutation positions
        if 'mutant' in df.columns:
            positions = [parse_mutant_string(m) for m in df['mutant']]
            scores = compute_masked_marginal_score(
                model, target_seq, sequences, positions, device, batch_size
            )
        else:
            # Fallback to log-likelihood
            scores = compute_log_likelihood(
                model, sequences, device, batch_size
            )
    else:
        scores = compute_log_likelihood(
            model, sequences, device, batch_size
        )

    df['proust_score'] = scores
    return df


def main():
    parser = argparse.ArgumentParser(description="Score ProteinGym with proust model")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to local checkpoint (downloads from HF if omitted)')
    parser.add_argument('--dms-file', type=str, default=None,
                        help='Path to single DMS CSV file')
    parser.add_argument('--dms-dir', type=str, default=None,
                        help='Path to directory of DMS CSV files')
    parser.add_argument('--reference-file', type=str, default=None,
                        help='Path to DMS reference file (for target sequences)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (for single file mode)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (for batch mode)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for scoring (larger = faster, more GPU memory)')
    parser.add_argument('--scoring-method', type=str, default='log_likelihood',
                        choices=['log_likelihood', 'masked_marginal'],
                        help='Scoring method')
    parser.add_argument('--indel-mode', action='store_true',
                        help='Enable indel mode')
    parser.add_argument('--dms-index', type=int, default=None,
                        help='Index of DMS to score (for array jobs)')
    args = parser.parse_args()

    # Load model
    model = load_model(local_path=args.checkpoint, device=args.device)

    # Load reference file if provided
    ref_df = None
    if args.reference_file:
        ref_df = pd.read_csv(args.reference_file)

    # Single file mode
    if args.dms_file:
        target_seq = None
        if ref_df is not None:
            dms_path = Path(args.dms_file)
            matches = ref_df[ref_df['DMS_filename'] == dms_path.name]
            if len(matches) == 0:
                matches = ref_df[ref_df['DMS_filename'].str.contains(dms_path.stem, na=False)]
            if len(matches) > 0:
                target_seq = matches.iloc[0]['target_seq']

        print(f"Scoring: {args.dms_file}")
        result_df = score_dms_file(
            model, args.dms_file, target_seq, args.device,
            args.batch_size, args.scoring_method, args.indel_mode
        )

        # Compute Spearman correlation
        if 'DMS_score' in result_df.columns:
            dms_scores = result_df['DMS_score'].to_numpy()
            proust_scores = result_df['proust_score'].to_numpy()
            mask = np.isfinite(dms_scores) & np.isfinite(proust_scores)
            spearman = spearmanr(dms_scores[mask], proust_scores[mask])[0] if mask.any() else np.nan
            print(f"Spearman correlation: {spearman:.4f} (n={int(mask.sum())})")

        output_path = args.output or args.dms_file.replace('.csv', '_scored.csv')
        result_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    # Batch mode
    elif args.dms_dir:
        if not args.output_dir:
            args.output_dir = args.dms_dir + '_scored'
        os.makedirs(args.output_dir, exist_ok=True)

        dms_files = sorted(Path(args.dms_dir).glob('*.csv'))

        # If dms_index specified, score only that file
        if args.dms_index is not None:
            dms_files = [dms_files[args.dms_index]]

        results = []
        for dms_file in tqdm(dms_files, desc="Scoring DMS files"):
            dms_name = dms_file.stem

            target_seq = None
            if ref_df is not None:
                matches = ref_df[ref_df['DMS_filename'] == dms_file.name]
                if len(matches) > 0:
                    target_seq = matches.iloc[0]['target_seq']

            try:
                result_df = score_dms_file(
                    model, str(dms_file), target_seq, args.device,
                    args.batch_size, args.scoring_method, args.indel_mode
                )

                # Save individual scores
                output_path = Path(args.output_dir) / f"{dms_name}.csv"
                result_df[['mutated_sequence', 'proust_score', 'DMS_score']].to_csv(output_path, index=False)

                # Compute correlation
                if 'DMS_score' in result_df.columns:
                    dms_scores = result_df['DMS_score'].to_numpy()
                    proust_scores = result_df['proust_score'].to_numpy()
                    mask = np.isfinite(dms_scores) & np.isfinite(proust_scores)
                    spearman = spearmanr(dms_scores[mask], proust_scores[mask])[0] if mask.any() else np.nan
                    results.append({
                        'DMS_id': dms_name,
                        'spearman': spearman,
                        'n_variants': int(mask.sum()),
                    })
                    print(f"{dms_name}: Spearman = {spearman:.4f} (n={int(mask.sum())})")
            except Exception as e:
                print(f"Error scoring {dms_name}: {e}")
                continue

        # Save summary
        if results:
            summary_df = pd.DataFrame(results)
            summary_path = Path(args.output_dir) / 'summary.csv'
            summary_df.to_csv(summary_path, index=False)

            mean_spearman = summary_df['spearman'].mean()
            print(f"\n=== Summary ===")
            print(f"Mean Spearman: {mean_spearman:.4f}")
            print(f"Scored {len(results)} DMS assays")
            print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
