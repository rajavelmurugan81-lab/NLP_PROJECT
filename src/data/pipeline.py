"""Main Data Ingestion & Preprocessing Pipeline."""

import os
import json
from typing import Dict, Any, List
from src.data.download import prepare_all_raw_datasets
from src.data.loader import get_loader
from src.data.preprocessor import preprocess_records
from src.data.validator import DatasetValidator, ValidationReport
from src.data.analyze import analyze_all_datasets


def process_dataset(dataset_name: str, processed_base_dir: str = "data/processed") -> Tuple[List[Dict[str, Any]], ValidationReport]:
    """Process a single dataset through loading, preprocessing, validation, and saving."""
    loader = get_loader(dataset_name)
    raw_records = loader.load()

    # Preprocess
    preprocessed_records, prep_stats = preprocess_records(raw_records)

    # Validate
    validator = DatasetValidator()
    val_report = validator.validate(preprocessed_records)

    # Save to processed directory
    out_dir = os.path.join(processed_base_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{dataset_name}_unified.jsonl")

    dicts = [r.to_dict() for r in preprocessed_records]

    with open(out_file, "w", encoding="utf-8") as f:
        for rec_dict in dicts:
            f.write(json.dumps(rec_dict) + "\n")

    print(f"[{dataset_name.upper()}] Ingestion Complete. {len(dicts)} records saved to {out_file}.")
    print(f"[{dataset_name.upper()}] Validation status: {'PASSED' if val_report.is_valid else 'WARNINGS/ERRORS'}")

    return dicts, val_report


def run_full_pipeline() -> Dict[str, Any]:
    """Execute complete Phase 3 Data Ingestion & Preprocessing Pipeline."""
    print("=== Phase 3: Data Ingestion & Preprocessing Pipeline ===")

    # 1. Download / Verify Raw Datasets
    prepare_all_raw_datasets()

    # 2. Ingest, preprocess, and validate datasets
    datasets = ["scifact", "covert", "sciclaimhunt"]
    pipeline_results = {}

    for ds in datasets:
        dicts, val_report = process_dataset(ds)
        pipeline_results[ds] = {
            "record_count": len(dicts),
            "validation_report": val_report.to_dict()
        }

    # 3. Generate analytics & statistics report
    stats = analyze_all_datasets()

    print("=== Phase 3 Pipeline Execution Finished Successfully ===")
    return {
        "pipeline_results": pipeline_results,
        "dataset_statistics": stats
    }


if __name__ == "__main__":
    run_full_pipeline()
