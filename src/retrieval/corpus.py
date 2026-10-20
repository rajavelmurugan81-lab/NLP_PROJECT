"""Corpus Builder & Document Index Construction module."""

import os
import json
from typing import List, Dict, Any, Tuple, Set
from src.data.schema import CorpusDocumentRecord, UnifiedClaimRecord
from src.data.loader import get_loader
from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG


class CorpusBuilder:
    """Builder class for assembling, deduplicating, and serializing the evidence search corpus."""

    def __init__(self, config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG):
        self.config = config

    def build_corpus_from_datasets(self, max_records_per_dataset: Dict[str, int] = None) -> Tuple[List[CorpusDocumentRecord], Dict[str, Any]]:
        """Construct the searchable document corpus from SciFact, CoVERT, and SciClaimHunt.

        Task 0 Safety Check Rules:
        - Total SciClaimHunt raw records: 110,002
        - SciClaimHunt missing claims: 1,109 (excluded from claim query sets)
        - SciClaimHunt valid records retained: 108,893
        """
        corpus_documents: List[CorpusDocumentRecord] = []
        seen_hashes: Set[str] = set()

        stats = {
            "total_raw_records": 0,
            "retained_documents": 0,
            "excluded_records": 0,
            "duplicates_removed": 0,
            "dataset_breakdown": {}
        }

        # 1. Process SciFact (Corpus abstracts + Rationale evidence)
        scifact_loader = get_loader("scifact")
        scifact_corpus = scifact_loader.load_corpus()  # Returns dict of doc_id -> CorpusDocumentRecord
        
        for doc_id, doc_record in scifact_corpus.items():
            corpus_documents.append(doc_record)
            stats["retained_documents"] += 1
            stats["total_raw_records"] += 1

        stats["dataset_breakdown"]["scifact"] = len(scifact_corpus)

        # 2. Process CoVERT
        try:
            covert_loader = get_loader("covert")
            covert_records = covert_loader.load()
            covert_added = 0
            for rec in covert_records:
                stats["total_raw_records"] += 1
                if not rec.evidence_text or not rec.evidence_text.strip():
                    stats["excluded_records"] += 1
                    continue
                
                doc_id = f"doc_covert_{rec.claim_id}"
                doc_rec = CorpusDocumentRecord(
                    doc_id=doc_id,
                    title=rec.evidence_title or "Biomedical Literature Context",
                    abstract=[rec.evidence_text],
                    source_dataset="covert",
                    structured=False,
                    metadata={"claim_id": rec.claim_id, "label": rec.label}
                )
                corpus_documents.append(doc_rec)
                covert_added += 1
                stats["retained_documents"] += 1
            stats["dataset_breakdown"]["covert"] = covert_added
        except Exception as e:
            print(f"[CorpusBuilder] Warning loading CoVERT: {e}")

        # 3. Process SciClaimHunt (Memory-efficient streaming)
        try:
            sch_loader = get_loader("sciclaimhunt")
            sch_added = 0
            sch_raw = 0
            max_sch = max_records_per_dataset.get("sciclaimhunt") if max_records_per_dataset else None

            for chunk in sch_loader.stream(chunk_size=5000):
                for rec in chunk:
                    sch_raw += 1
                    stats["total_raw_records"] += 1

                    # Task 0 Safety Check Rule:
                    # Exclusion Rule: Exclude records where claim is empty OR evidence_text is empty
                    if not rec.claim or not rec.claim.strip():
                        stats["excluded_records"] += 1
                        continue

                    if not rec.evidence_text or not rec.evidence_text.strip():
                        stats["excluded_records"] += 1
                        continue

                    # Exact deduplication check based on evidence text hash
                    ev_text = rec.evidence_text.strip()
                    ev_hash = hash((ev_text[:200], rec.source_dataset))
                    if ev_hash in seen_hashes:
                        stats["duplicates_removed"] += 1
                        continue

                    seen_hashes.add(ev_hash)
                    doc_id = f"doc_sch_{rec.claim_id}"
                    doc_rec = CorpusDocumentRecord(
                        doc_id=doc_id,
                        title=rec.evidence_title or "Research Paper Section",
                        abstract=[ev_text],
                        source_dataset="sciclaimhunt",
                        structured=False,
                        metadata={"claim_id": rec.claim_id, "label": rec.label}
                    )
                    corpus_documents.append(doc_rec)
                    sch_added += 1
                    stats["retained_documents"] += 1

                    if max_sch and sch_added >= max_sch:
                        break
                if max_sch and sch_added >= max_sch:
                    break

            stats["dataset_breakdown"]["sciclaimhunt"] = sch_added
        except Exception as e:
            print(f"[CorpusBuilder] Warning loading SciClaimHunt: {e}")

        return corpus_documents, stats

    def save_corpus(self, corpus_documents: List[CorpusDocumentRecord], filepath: str = None) -> str:
        """Save evidence corpus to unified JSONL format."""
        target_path = filepath or self.config.corpus_file
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            for doc in corpus_documents:
                f.write(json.dumps(doc.to_dict()) + "\n")

        print(f"[CorpusBuilder] Saved {len(corpus_documents):,} corpus records to {target_path}")
        return target_path

    def load_corpus(self, filepath: str = None) -> List[CorpusDocumentRecord]:
        """Load evidence corpus from unified JSONL format."""
        target_path = filepath or self.config.corpus_file
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"[CorpusBuilder] Corpus file not found at {target_path}")

        records = []
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(CorpusDocumentRecord.from_dict(data))
        return records
