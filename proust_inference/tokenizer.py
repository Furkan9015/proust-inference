"""ESM-style protein tokenizer (vocab_size=32)."""

import torch

ESM_TOKENS = {
    '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8,
    'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13,
    'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18,
    'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23,
    'B': 24, 'U': 25, 'Z': 26, 'O': 27, 'X': 28,
}

ID_TO_TOKEN = {v: k for k, v in ESM_TOKENS.items()}


def tokenize(
    sequence: str,
    add_special_tokens: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Tokenize a protein sequence.

    Args:
        sequence: Amino acid string (e.g. "MKTLLILAVL")
        add_special_tokens: Prepend <cls> and append <eos>
        device: Target device

    Returns:
        1D int64 tensor of token IDs
    """
    ids = []
    if add_special_tokens:
        ids.append(ESM_TOKENS['<cls>'])
    for aa in sequence.upper():
        ids.append(ESM_TOKENS.get(aa, ESM_TOKENS['<unk>']))
    if add_special_tokens:
        ids.append(ESM_TOKENS['<eos>'])
    return torch.tensor(ids, dtype=torch.long, device=device)


def tokenize_batch(
    sequences: list[str],
    add_special_tokens: bool = True,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, list[int]]:
    """Tokenize and pad a batch of sequences.

    Returns:
        (padded_ids, lengths) where padded_ids is (batch, max_len)
    """
    tokenized = [tokenize(s, add_special_tokens, device="cpu") for s in sequences]
    lengths = [len(t) for t in tokenized]
    max_len = max(lengths)
    padded = torch.full((len(sequences), max_len), ESM_TOKENS['<pad>'], dtype=torch.long)
    for i, t in enumerate(tokenized):
        padded[i, :len(t)] = t
    return padded.to(device), lengths


def decode(token_ids: torch.Tensor) -> str:
    """Decode token IDs back to amino acid string (strips special tokens)."""
    chars = []
    for tid in token_ids.tolist():
        tok = ID_TO_TOKEN.get(tid, '?')
        if tok in ('<cls>', '<pad>', '<eos>', '<unk>'):
            continue
        chars.append(tok)
    return ''.join(chars)
