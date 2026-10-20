"""Unit tests for ExperimentRunner end-to-end execution and result persistence."""

import os
import tempfile
import pytest

from src.evaluation.runner import ExperimentRunner


def test_experiment_runner_dev_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "test_eval_config.yaml")
        exp_dir = os.path.join(tmp_dir, "results/experiments")

        config_data = {
            "experiment": {
                "name": "test_run",
                "output_dir": exp_dir
            },
            "system": {
                "provider_type": "mock"
            },
            "ledger": {
                "ledger_path": os.path.join(tmp_dir, "test_ledger.jsonl")
            }
        }

        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        runner = ExperimentRunner(config_filepath=config_path)
        result = runner.run_experiment(
            system_name="citeguard_hybrid_multiagent",
            dataset="scifact",
            split="test",
            limit=5,
            run_id_override="test_dev_run_01"
        )

        assert result.run_id == "test_dev_run_01"
        assert result.total_examples == 5
        assert result.classification_metrics is not None
        assert result.metadata["development_run"] is True

        # Verify output directory and artifacts
        run_dir = os.path.join(exp_dir, "test_dev_run_01")
        assert os.path.exists(os.path.join(run_dir, "config.yaml"))
        assert os.path.exists(os.path.join(run_dir, "predictions.jsonl"))
        assert os.path.exists(os.path.join(run_dir, "metrics.json"))
        assert os.path.exists(os.path.join(run_dir, "errors.jsonl"))
        assert os.path.exists(os.path.join(run_dir, "metadata.json"))
