"""Query-side tokenizer.

This must stay identical to ``normalize_tokens`` in
``scripts/preprocess_proposals.py``, which produces the ``tokens`` field the
index is built from. If the team adds stemming there, add it here too.
"""

from pathlib import Path
from typing import FrozenSet, Iterable, List, Optional
import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

DEFAULT_STOPWORDS_PATH = Path(__file__).resolve().parents[2] / "stopwords.txt"


def load_stopwords(path: Optional[Path] = None) -> FrozenSet[str]:
    """Load the shared stopword list (one term per line)."""
    path = Path(path) if path is not None else DEFAULT_STOPWORDS_PATH
    if not path.exists():
        return frozenset()
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip().lower() for line in lines if line.strip())


def tokenize(text: str, stopwords: Iterable[str] = ()) -> List[str]:
    """Lowercase, split into word tokens, and drop stopwords."""
    if not isinstance(stopwords, (set, frozenset)):
        stopwords = frozenset(stopwords)
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in stopwords]
