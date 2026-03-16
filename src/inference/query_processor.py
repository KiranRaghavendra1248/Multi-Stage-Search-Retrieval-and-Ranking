import re
import nltk
from functools import lru_cache
from spellchecker import SpellChecker
from omegaconf import DictConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Stopwords to skip during synonym expansion
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "through",
    "and", "or", "but", "if", "then", "than", "so", "yet", "both",
    "not", "no", "nor", "that", "this", "these", "those", "it", "its",
}

# POS tags we expand (nouns and verbs only)
_EXPAND_POS = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}


@lru_cache(maxsize=1)
def _get_spell_checker() -> SpellChecker:
    return SpellChecker()


@lru_cache(maxsize=4096)
def _get_synonym(word: str) -> str | None:
    """Return the most common synonym for word from WordNet, or None."""
    from nltk.corpus import wordnet
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    # Lemmas of the first synset, sorted by frequency descending
    lemmas = sorted(
        synsets[0].lemmas(),
        key=lambda l: -l.count()
    )
    for lemma in lemmas:
        candidate = lemma.name().replace("_", " ")
        if candidate.lower() != word.lower():
            return candidate
    return None


def correct_spelling(text: str) -> str:
    """Correct spelling mistakes word-by-word."""
    checker = _get_spell_checker()
    words = text.split()
    corrected = []
    for word in words:
        # Preserve words with digits, URLs, or capitalization (proper nouns)
        if re.search(r"[0-9]", word) or word[0].isupper():
            corrected.append(word)
            continue
        fixed = checker.correction(word.lower())
        corrected.append(fixed if fixed else word)
    return " ".join(corrected)


def expand_synonyms(text: str) -> str:
    """
    Append synonyms for content words (nouns + verbs, non-stopwords).

    Example:
        "car speed limit" → "car speed limit automobile velocity restriction"
    """
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    additions = []
    for word, pos in tagged:
        if pos not in _EXPAND_POS:
            continue
        if word.lower() in _STOPWORDS:
            continue
        syn = _get_synonym(word.lower())
        if syn and syn.lower() != word.lower():
            additions.append(syn)

    if additions:
        return text + " " + " ".join(additions)
    return text


def process_query(query: str, cfg: DictConfig) -> str:
    """
    Pre-retrieval query processing pipeline.

    Steps (only when cfg.inference.query_rewriting == True):
        1. Spell correction
        2. Synonym expansion (nouns + verbs only, max 1 synonym per word)

    Returns the (possibly rewritten) query string.
    """
    if not cfg.inference.get("query_rewriting", False):
        return query

    corrected = correct_spelling(query)
    expanded = expand_synonyms(corrected)

    if expanded != query:
        logger.debug("Query rewritten: %r → %r", query, expanded)

    return expanded
