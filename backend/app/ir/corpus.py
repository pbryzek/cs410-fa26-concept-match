"""Load the preprocessed proposal corpus written by scripts/preprocess_proposals.py."""

from pathlib import Path
from typing import Dict, List, Tuple
import json


def load_corpus(path) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
    """Return ({submission_id: tokens}, {submission_id: full record})."""
    docs: Dict[str, List[str]] = {}
    records: Dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record["submission_id"]
            docs[doc_id] = record.get("tokens") or []
            records[doc_id] = record
    return docs, records
