# LDA Topic Modeling and Concept Tagging

This pipeline trains a Latent Dirichlet Allocation (LDA) topic model using `tomotopy` to automatically infer canonical CS410 concept tags for past proposals and resolve vocabulary mismatches.

---

## End-to-End Workflow

### Step 1: Export Proposals from Snapshot
Exports raw proposals from the snapshot into normalized JSONL format.
```bash
python3 scripts/export_proposals.py
```

### Step 2: Preprocess Text
Cleans text, strips stopwords, and produces tokenized proposal records.
```bash
python3 scripts/preprocess_proposals.py
```

### Step 3: (Optional) Evaluate Topic Coherence
Tests candidate topic counts to find the best number of topics using coherence scores.
```bash
python3 scripts/train_lda.py --evaluate-k --k-candidates 5,7,10,14,18
```

### Step 4: Train LDA Model and Tag Proposals
Trains the final model on preprocessed proposals and outputs the tagged dataset with inferred concept tags.
```bash
python3 scripts/train_lda.py --k 10
```
* Generates `data/processed/lda_model.bin` (trained model binary).
* Generates `data/processed/proposals_tagged.jsonl` (proposals enriched with `inferred_concept_tags`).

### Step 5: (Optional) Test Query Concept Tagging (Dev Only)
CLI tool for development and testing purposes only to verify how search queries map to canonical CS410 concept tags.
```bash
python3 scripts/infer_tags.py --text "Rocchio"
```

---

## Output Files

| File | Description |
| :--- | :--- |
| `data/processed/proposals_preprocessed.jsonl` | Cleaned and tokenized proposal corpus. |
| `data/processed/lda_model.bin` | Serialized tomotopy LDA model binary. |
| `data/processed/proposals_tagged.jsonl` | Final proposals with `inferred_concept_tags` for downstream BM25 indexing. |
