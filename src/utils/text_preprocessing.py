"""Text preprocessing utilities: normalization, tokenization, vocabulary building."""

import re
import string
from collections import Counter
from typing import Dict, List, Optional


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = False,
    remove_punctuation: bool = False,
    expand_contractions: bool = True,
    remove_emojis: bool = False,
) -> str:
    """Normalize raw text with configurable cleaning steps.

    Args:
        text: Input string.
        lowercase: Convert to lowercase.
        remove_urls: Strip http/https URLs.
        remove_emails: Strip email addresses.
        remove_numbers: Replace digits with empty string.
        remove_punctuation: Strip punctuation characters.
        expand_contractions: Expand common English contractions.
        remove_emojis: Remove emoji characters.

    Returns:
        Cleaned string.
    """
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
    if remove_emails:
        text = re.sub(r"\S+@\S+\.\S+", "", text)
    if expand_contractions:
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'s": " is", "'d": " would",
            "'ll": " will", "'ve": " have", "'m": " am",
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
    if lowercase:
        text = text.lower()
    if remove_emojis:
        emoji_pattern = re.compile(
            "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF" "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)
    if remove_numbers:
        text = re.sub(r"\d+", "", text)
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def basic_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Simple whitespace + punctuation-aware tokenizer.

    Splits on whitespace and separates punctuation as individual tokens.
    """
    if lowercase:
        text = text.lower()
    # Insert space before/after punctuation so split captures them
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    return text.split()


def build_vocab(
    texts: List[str],
    max_vocab_size: Optional[int] = None,
    min_freq: int = 1,
    special_tokens: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Build a word-to-index vocabulary from tokenized texts.

    Args:
        texts: List of raw text strings (will be tokenized internally).
        max_vocab_size: Cap the vocabulary size (excluding special tokens).
        min_freq: Minimum token frequency to include.
        special_tokens: Tokens like ['<PAD>', '<UNK>'] added at the start.

    Returns:
        Dict mapping token -> index.
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>"]

    counter = Counter()
    for text in texts:
        counter.update(basic_tokenize(text))

    # Filter by min_freq
    filtered = {tok: cnt for tok, cnt in counter.items() if cnt >= min_freq}

    # Sort by frequency descending
    sorted_tokens = sorted(filtered, key=lambda t: filtered[t], reverse=True)

    if max_vocab_size is not None:
        sorted_tokens = sorted_tokens[:max_vocab_size]

    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for tok in sorted_tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


def texts_to_sequences(
    texts: List[str],
    vocab: Dict[str, int],
    max_len: Optional[int] = None,
) -> List[List[int]]:
    """Convert texts to padded integer sequences using a vocabulary.

    Args:
        texts: List of raw strings.
        vocab: Token-to-index mapping (must contain '<PAD>' and '<UNK>').
        max_len: If set, truncate/pad sequences to this length.

    Returns:
        List of integer sequences.
    """
    pad_idx = vocab.get("<PAD>", 0)
    unk_idx = vocab.get("<UNK>", 1)

    sequences = []
    for text in texts:
        tokens = basic_tokenize(text)
        seq = [vocab.get(t, unk_idx) for t in tokens]
        if max_len is not None:
            seq = seq[:max_len]
            seq = seq + [pad_idx] * (max_len - len(seq))
        sequences.append(seq)
    return sequences
