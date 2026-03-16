import re
from functools import lru_cache
from transformers import AutoTokenizer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Header patterns: short line (≤ 80 chars) that looks like a section title
_HEADER_RE = re.compile(r"^.{1,80}$")
# Horizontal rules
_HR_RE = re.compile(r"^[-=*]{3,}\s*$")


@lru_cache(maxsize=1)
def _get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def _token_count(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _is_section_boundary(line: str, next_line: str) -> bool:
    """True if line looks like a section header (short + followed by blank line)."""
    return bool(_HEADER_RE.match(line.strip())) and next_line.strip() == ""


def _split_paragraph(paragraph: str, tokenizer, max_tokens: int) -> list[str]:
    """Fall back to sentence-level splitting for oversized paragraphs."""
    import nltk
    sentences = nltk.sent_tokenize(paragraph)
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        sent_tokens = _token_count(sent, tokenizer)
        if current and current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += sent_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    text: str,
    tokenizer_name: str = "bert-base-uncased",
    max_tokens: int = 256,
    min_tokens_merge: int = 64,
) -> list[str]:
    """
    Paragraph-aware semantic chunking.

    Algorithm:
    1. Split on paragraph/section boundaries (\n\n, ---, header patterns).
    2. Each paragraph ≤ max_tokens → emit as one chunk.
    3. Paragraph > max_tokens → sentence-level fallback splitting.
    4. Consecutive paragraphs < min_tokens_merge → merge with next to avoid tiny chunks.

    Returns a list of passage strings.
    """
    tokenizer = _get_tokenizer(tokenizer_name)

    # --- Step 1: split into raw paragraph blocks ---
    lines = text.splitlines()
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        if _HR_RE.match(line):
            if current_lines:
                paragraphs.append("\n".join(current_lines).strip())
                current_lines = []
            continue

        if line.strip() == "":
            if current_lines:
                paragraphs.append("\n".join(current_lines).strip())
                current_lines = []
            continue

        if _is_section_boundary(line, next_line):
            if current_lines:
                paragraphs.append("\n".join(current_lines).strip())
            paragraphs.append(line.strip())
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        paragraphs.append("\n".join(current_lines).strip())

    paragraphs = [p for p in paragraphs if p]

    # --- Step 2 & 3: per-paragraph chunking ---
    raw_chunks: list[str] = []
    for para in paragraphs:
        count = _token_count(para, tokenizer)
        if count <= max_tokens:
            raw_chunks.append(para)
        else:
            raw_chunks.extend(_split_paragraph(para, tokenizer, max_tokens))

    # --- Step 4: merge tiny chunks ---
    merged: list[str] = []
    i = 0
    while i < len(raw_chunks):
        chunk = raw_chunks[i]
        while (
            i + 1 < len(raw_chunks)
            and _token_count(chunk, tokenizer) < min_tokens_merge
            and _token_count(chunk + " " + raw_chunks[i + 1], tokenizer) <= max_tokens
        ):
            i += 1
            chunk = chunk + " " + raw_chunks[i]
        merged.append(chunk)
        i += 1

    return merged
