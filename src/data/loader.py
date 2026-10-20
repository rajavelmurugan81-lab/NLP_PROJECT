"""Modular Dataset Loaders for SciFact, CoVERT, and SciClaimHunt."""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from src.data.schema import UnifiedClaimRecord, CorpusDocumentRecord


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, dataset_name: str, raw_dir: str):
        self.dataset_name = dataset_name.lower()
        self.raw_dir = raw_dir

    @abstractmethod
    def load(self) -> List[UnifiedClaimRecord]:
        """Load dataset records and convert to unified schema."""
        pass


class SciFactLoader(BaseDatasetLoader):
    """Loader for the SciFact dataset."""

    def __init__(self, raw_dir: str = "data/raw/scifact"):
        super().__init__("scifact", raw_dir)
        # Handle nested data directory extracted from data.tar.gz
        if os.path.exists(os.path.join(raw_dir, "data")):
            self.data_dir = os.path.join(raw_dir, "data")
        else:
            self.data_dir = raw_dir

    def load_corpus(self) -> Dict[str, CorpusDocumentRecord]:
        """Load PubMed abstract corpus from corpus.jsonl."""
        corpus_path = os.path.join(self.data_dir, "corpus.jsonl")
        corpus = {}
        if not os.path.exists(corpus_path):
            print(f"[SciFactLoader] Corpus file not found at {corpus_path}")
            return corpus

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc_id = str(data["doc_id"])
                    corpus[doc_id] = CorpusDocumentRecord(
                        doc_id=doc_id,
                        title=str(data.get("title", "")),
                        abstract=list(data.get("abstract", [])),
                        source_dataset="scifact",
                        structured=bool(data.get("structured", False)),
                        metadata={"structured": data.get("structured", False)}
                    )
                except KeyError as e:
                    raise ValueError(f"[SciFactLoader] Corpus line {line_no} missing required field: {e}")
                except json.JSONDecodeError:
                    raise ValueError(f"[SciFactLoader] Corrupted JSON on line {line_no} in {corpus_path}")
        return corpus

    def load(self) -> List[UnifiedClaimRecord]:
        """Load claim records from train, dev, and test jsonl files."""
        corpus = self.load_corpus()
        records: List[UnifiedClaimRecord] = []

        splits = [
            ("train", "claims_train.jsonl"),
            ("dev", "claims_dev.jsonl"),
            ("test", "claims_test.jsonl")
        ]

        for split_name, filename in splits:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"[SciFactLoader] Warning: Split file {filepath} not found.")
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"[SciFactLoader] Corrupted JSON on line {line_no} in {filepath}")

                    if "id" not in raw or "claim" not in raw:
                        raise ValueError(f"[SciFactLoader] Record on line {line_no} missing 'id' or 'claim'")

                    claim_id = f"scifact_{raw['id']}"
                    claim_text = str(raw["claim"]).strip()
                    evidence_dict = raw.get("evidence", {})
                    cited_doc_ids = [str(d) for d in raw.get("cited_doc_ids", [])]

                    if not evidence_dict:
                        # Neutral / Unsupported claim or unannotated test set
                        records.append(UnifiedClaimRecord(
                            claim_id=claim_id,
                            claim=claim_text,
                            label="NEUTRAL",
                            source_dataset="scifact",
                            split=split_name,
                            evidence_id=cited_doc_ids[0] if cited_doc_ids else None,
                            evidence_text=None,
                            evidence_title=corpus[cited_doc_ids[0]].title if (cited_doc_ids and cited_doc_ids[0] in corpus) else None,
                            metadata={"cited_doc_ids": cited_doc_ids, "original_id": raw["id"]}
                        ))
                    else:
                        for doc_id, ev_list in evidence_dict.items():
                            doc_id_str = str(doc_id)
                            doc_obj = corpus.get(doc_id_str)
                            doc_title = doc_obj.title if doc_obj else None

                            for ev_idx, ev in enumerate(ev_list):
                                sentence_indices = ev.get("sentences", [])
                                label_raw = str(ev.get("label", "NEUTRAL")).upper()

                                # Map label
                                if label_raw in ["SUPPORT", "SUPPORTS"]:
                                    label = "SUPPORT"
                                elif label_raw in ["CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES"]:
                                    label = "REFUTE"
                                else:
                                    label = "NEUTRAL"

                                ev_sentences = []
                                if doc_obj and doc_obj.abstract:
                                    for idx in sentence_indices:
                                        if 0 <= idx < len(doc_obj.abstract):
                                            ev_sentences.append(doc_obj.abstract[idx])

                                ev_text = " ".join(ev_sentences) if ev_sentences else None

                                sub_id = f"{claim_id}_doc{doc_id_str}_ev{ev_idx}"
                                records.append(UnifiedClaimRecord(
                                    claim_id=sub_id,
                                    claim=claim_text,
                                    label=label,
                                    source_dataset="scifact",
                                    split=split_name,
                                    evidence_id=doc_id_str,
                                    evidence_text=ev_text,
                                    evidence_title=doc_title,
                                    metadata={
                                        "original_claim_id": raw["id"],
                                        "sentence_indices": sentence_indices,
                                        "cited_doc_ids": cited_doc_ids
                                    }
                                ))

        return records


class CoVERTLoader(BaseDatasetLoader):
    """Loader for the CoVERT biomedical dataset."""

    def __init__(self, raw_dir: str = "data/raw/covert"):
        super().__init__("covert", raw_dir)

    def load(self) -> List[UnifiedClaimRecord]:
        filepath = os.path.join(self.raw_dir, "covert.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[CoVERTLoader] File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"[CoVERTLoader] Invalid JSON in {filepath}")

        if not isinstance(raw_data, list):
            raise ValueError(f"[CoVERTLoader] Expected a JSON list of records in {filepath}")

        records: List[UnifiedClaimRecord] = []
        for idx, item in enumerate(raw_data):
            if "claim" not in item and "tweet" not in item:
                raise ValueError(f"[CoVERTLoader] Item index {idx} missing 'claim' or 'tweet'")

            claim_id = str(item.get("claim_id", item.get("id", f"covert_{idx:03d}")))
            claim_text = str(item.get("claim", item.get("tweet", ""))).strip()
            
            raw_label = str(item.get("verdict", item.get("label", "NEUTRAL"))).upper()
            if raw_label in ["SUPPORT", "SUPPORTS", "SUPPORTED"]:
                label = "SUPPORT"
            elif raw_label in ["REFUTE", "REFUTES", "REFUTED", "CONTRADICT"]:
                label = "REFUTE"
            elif raw_label in ["PARTIALLY_SUPPORTED", "PARTIAL"]:
                label = "PARTIALLY_SUPPORTED"
            else:
                label = "NEUTRAL"

            ev_text = item.get("evidence")
            if isinstance(ev_text, list):
                ev_text = " ".join(ev_text)

            split = str(item.get("split", "train")).lower()

            records.append(UnifiedClaimRecord(
                claim_id=claim_id,
                claim=claim_text,
                label=label,
                source_dataset="covert",
                split=split,
                evidence_id=str(item.get("evidence_id", f"ev_{claim_id}")),
                evidence_text=str(ev_text).strip() if ev_text else None,
                evidence_title=str(item.get("evidence_title", "Biomedical Literature Context")),
                metadata=dict(item.get("metadata", {}))
            ))

        return records


class SciClaimHuntLoader(BaseDatasetLoader):
    """Loader for the SciClaimHunt dataset supporting memory-efficient streaming."""

    def __init__(self, raw_dir: str = "data/raw/sciclaimhunt"):
        super().__init__("sciclaimhunt", raw_dir)

    def stream(self, chunk_size: int = 1000):
        """Yield records in memory-efficient chunks generator to avoid high RAM usage."""
        csv_filepath = os.path.join(self.raw_dir, "dataset.csv")

        if os.path.exists(csv_filepath):
            import csv
            import sys
            try:
                csv.field_size_limit(sys.maxsize)
            except OverflowError:
                csv.field_size_limit(2147483647)

            with open(csv_filepath, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                chunk = []
                for idx, row in enumerate(reader):
                    claim_text = str(row.get("Claim", "")).strip()
                    if not claim_text:
                        continue

                    raw_type = str(row.get("Type", "")).lower()
                    if "positive" in raw_type or "support" in raw_type:
                        label = "SUPPORT"
                    elif "negative" in raw_type or "refute" in raw_type:
                        label = "REFUTE"
                    else:
                        label = "NEUTRAL"

                    ev_text = str(row.get("Evidence", "")).strip()
                    claim_id = f"sch_csv_{idx}"

                    rec = UnifiedClaimRecord(
                        claim_id=claim_id,
                        claim=claim_text,
                        label=label,
                        source_dataset="sciclaimhunt",
                        split="train",
                        evidence_id=f"ev_{claim_id}",
                        evidence_text=ev_text if ev_text else None,
                        evidence_title="Full Text Research Paper",
                        metadata={"full_paper_snippet": str(row.get("research_paper_full", ""))[:500]}
                    )
                    chunk.append(rec)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk
            return

        # Fallback to fixture jsonl files
        records = self._load_fixtures()
        for i in range(0, len(records), chunk_size):
            yield records[i:i + chunk_size]

    def _load_fixtures(self) -> List[UnifiedClaimRecord]:
        """Load fixture JSONL files from fixtures/ or raw_dir."""
        records: List[UnifiedClaimRecord] = []
        splits = ["train", "dev", "test"]

        # Check fixtures directory first, then raw_dir
        search_dirs = [os.path.join(self.raw_dir, "fixtures"), self.raw_dir]

        for split_name in splits:
            filepath = None
            for sdir in search_dirs:
                cand = os.path.join(sdir, f"sciclaimhunt_{split_name}.jsonl")
                if os.path.exists(cand):
                    filepath = cand
                    break

            if not filepath or not os.path.exists(filepath):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"[SciClaimHuntLoader] Corrupted JSON line {line_no} in {filepath}")

                    if "claim" not in item:
                        raise ValueError(f"[SciClaimHuntLoader] Item line {line_no} missing required 'claim' field")

                    claim_id = str(item.get("claim_id", f"sch_{split_name}_{line_no}"))
                    claim_text = str(item["claim"]).strip()

                    raw_label = str(item.get("label", "NEUTRAL")).upper()
                    if raw_label in ["SUPPORT", "SUPPORTS", "SUPPORTED"]:
                        label = "SUPPORT"
                    elif raw_label in ["REFUTE", "REFUTES", "REFUTED", "CONTRADICT"]:
                        label = "REFUTE"
                    elif raw_label in ["PARTIALLY_SUPPORTED", "PARTIAL"]:
                        label = "PARTIALLY_SUPPORTED"
                    else:
                        label = "NEUTRAL"

                    ev_spans = item.get("evidence_spans", [])
                    ev_text = " ".join(ev_spans) if isinstance(ev_spans, list) else str(ev_spans)

                    records.append(UnifiedClaimRecord(
                        claim_id=claim_id,
                        claim=claim_text,
                        label=label,
                        source_dataset="sciclaimhunt",
                        split=split_name,
                        evidence_id=str(item.get("evidence_id", f"ev_{claim_id}")),
                        evidence_text=ev_text.strip() if ev_text else None,
                        evidence_title=str(item.get("section_name", "Paper Section")),
                        metadata={
                            "section_name": item.get("section_name"),
                            "raw_metadata": item.get("metadata", {})
                        }
                    ))

        return records

    def load(self, max_records: Optional[int] = None) -> List[UnifiedClaimRecord]:
        """Load dataset records into memory (optionally capped via max_records)."""
        records: List[UnifiedClaimRecord] = []
        for chunk in self.stream(chunk_size=5000):
            records.extend(chunk)
            if max_records and len(records) >= max_records:
                return records[:max_records]
        return records


def get_loader(dataset_name: str, raw_dir: Optional[str] = None) -> BaseDatasetLoader:
    """Factory function to retrieve dataset loader instance."""
    name = dataset_name.lower()
    if name == "scifact":
        return SciFactLoader(raw_dir=raw_dir or "data/raw/scifact")
    elif name == "covert":
        return CoVERTLoader(raw_dir=raw_dir or "data/raw/covert")
    elif name in ["sciclaimhunt", "sciclaimhunt_num"]:
        return SciClaimHuntLoader(raw_dir=raw_dir or "data/raw/sciclaimhunt")
    else:
        raise ValueError(f"Unknown dataset loader requested: {dataset_name}")
