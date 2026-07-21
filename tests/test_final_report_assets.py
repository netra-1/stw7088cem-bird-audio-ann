from __future__ import annotations

import csv
import hashlib
import inspect
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from bird_audio import final_report_assets as report
from bird_audio.config import LOCKED_TASK1_CLASS_ORDER

SOURCE_SHA256 = "a" * 64
SPECIES = (
    "Acridotheres fuscus",
    "Ceryle rudis",
    "Corvus splendens",
    "Ortygornis pondicerianus",
    "Psilopogon zeylanicus",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _seed_summary(metric: str, values: tuple[float, float, float]) -> dict[str, object]:
    mean = sum(values) / 3.0
    sample_variance = sum((value - mean) ** 2 for value in values) / 2.0
    return {
        "metric_name": metric,
        "seeds": [13, 37, 71],
        "values": list(values),
        "mean": mean,
        "sample_standard_deviation": sample_variance**0.5,
        "standard_deviation_ddof": 1,
    }


def _interval(point: float) -> dict[str, float]:
    return {
        "lower": max(0.0, point - 0.05),
        "upper": min(1.0, point + 0.05),
        "confidence_level": 0.95,
    }


def _task1_summary() -> dict[str, object]:
    count = len(LOCKED_TASK1_CLASS_ORDER)
    confusion = [[1 if row == column else 0 for column in range(count)] for row in range(count)]
    return {
        "stability": {
            "seeds": [13, 37, 71],
            "accuracy": _seed_summary("accuracy", (0.80, 0.82, 0.81)),
            "macro_f1": _seed_summary("macro_f1", (0.78, 0.81, 0.79)),
        },
        "seed_37_metrics": {
            "recording_count": count,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "class_order": list(LOCKED_TASK1_CLASS_ORDER),
            "per_class": [
                {
                    "class_index": index,
                    "class_name": name,
                    "support": 1,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                }
                for index, name in enumerate(LOCKED_TASK1_CLASS_ORDER)
            ],
            "confusion_counts": confusion,
            "row_normalized_confusion": [[float(value) for value in row] for row in confusion],
            "zero_division": 0,
        },
        "seed_37_bootstrap": {
            "task1_seed": 37,
            "accuracy_interval": _interval(0.82),
            "macro_f1_interval": _interval(0.81),
            "per_class_f1_intervals": [
                {
                    "class_index": index,
                    "class_name": name,
                    **_interval(0.9),
                }
                for index, name in enumerate(LOCKED_TASK1_CLASS_ORDER)
            ],
        },
    }


def _metric_values(base: float) -> dict[str, float]:
    return {
        "auroc": base,
        "sensitivity": base - 0.02,
        "specificity": base + 0.02,
        "balanced_accuracy": base,
    }


def _metric_summaries(base: float) -> list[dict[str, object]]:
    offsets = {"auroc": 0.0, "sensitivity": -0.02, "specificity": 0.02, "balanced_accuracy": 0.0}
    return [
        _seed_summary(
            metric,
            (base + offsets[metric] - 0.01, base + offsets[metric], base + offsets[metric] + 0.01),
        )
        for metric in report.TASK2_METRIC_ORDER
    ]


def _metric_intervals(base: float) -> dict[str, dict[str, float]]:
    return {metric: _interval(value) for metric, value in _metric_values(base).items()}


def _task2_stream(base: float, score_name: str) -> dict[str, object]:
    per_species_points = []
    per_species_intervals = []
    per_species_stability = []
    for index, species in enumerate(SPECIES):
        value = base - 0.04 + index * 0.02
        per_species_points.append(
            {
                "species_scientific_name": species,
                **_metric_values(value),
                "known_recording_count": 267,
                "unknown_recording_count": 40,
            }
        )
        per_species_intervals.append(
            {"species_scientific_name": species, **_metric_intervals(value)}
        )
        per_species_stability.append(
            {
                "species_scientific_name": species,
                "metrics": _metric_summaries(value),
            }
        )
    point = {
        "threshold": 0.3,
        "pooled": {
            **_metric_values(base),
            "known_recording_count": 267,
            "unknown_recording_count": 200,
        },
        "per_species": per_species_points,
        "macro": _metric_values(base),
    }
    return {
        "score_name": score_name,
        "stability": {
            "seed_order": [13, 37, 71],
            "pooled": _metric_summaries(base),
            "per_species": per_species_stability,
            "macro": _metric_summaries(base),
        },
        "seed_37_point_estimates": point,
        "seed_37_bootstrap": {
            "point_estimates": point,
            "pooled_intervals": _metric_intervals(base),
            "per_species_intervals": per_species_intervals,
            "macro_intervals": _metric_intervals(base),
        },
    }


class ReportFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name).resolve() / "project"
        self.project.mkdir()
        self.report_root = self.project / "report_assets" / "final_v2"
        self.attempt = self.project / "runs" / "final_evaluation_v2" / "attempt_v2"
        self.gate_root = self.project / "runs" / "final_evaluation_v2" / "gate_v2"
        self.final_result_record = _write_json(self.attempt / "result.json", {"final": True})
        self.final_lock_record = _write_json(self.attempt / "lock.json", {"locked": True})
        self.final_result_record["path"] = "result.json"
        self.final_lock_record["path"] = "lock.json"
        self.gate_record = _write_json(self.gate_root / "gate.json", {"gate": True})
        self.gate_lock_record = _write_json(self.gate_root / "lock.json", {"locked": True})
        self.gate = {
            "shared_identity": {"source_fingerprint_sha256": SOURCE_SHA256},
            "task1": {"runs": self._runs("task1")},
            "task2": {"runs": self._runs("task2")},
        }
        self.final = {
            "source_fingerprint_sha256": SOURCE_SHA256,
            "gate_sha256": self.gate_record["sha256"],
            "seed_order": [13, 37, 71],
            "task1_summary": _task1_summary(),
            "task2_summary": {
                "reconstruction": _task2_stream(0.82, "median_clip_reconstruction_mse"),
                "latent": _task2_stream(0.76, "recording_mean_latent_knn_distance"),
            },
            "result_artifact": self.final_result_record,
            "completion_lock_artifact": self.final_lock_record,
        }
        self.final_calls = 0
        self.gate_calls = 0
        self.stack = ExitStack()
        self.stack.enter_context(
            mock.patch.multiple(
                report,
                PROJECT_ROOT=self.project,
                FINAL_REPORT_ASSET_ROOT=self.report_root,
                FINAL_REPORT_MANIFEST_PATH=self.report_root / "manifest.json",
                FINAL_REPORT_LOCK_PATH=self.report_root / "lock.json",
                FINAL_EVALUATION_ATTEMPT_DIRECTORY=self.attempt,
                FINAL_EVALUATION_RESULT_PATH=self.attempt / "result.json",
                FINAL_EVALUATION_LOCK_PATH=self.attempt / "lock.json",
                FINAL_EVALUATION_GATE_PATH=self.gate_root / "gate.json",
                FINAL_EVALUATION_GATE_LOCK_PATH=self.gate_root / "lock.json",
            )
        )
        self.stack.enter_context(
            mock.patch.object(report, "source_fingerprint", return_value=SOURCE_SHA256)
        )
        self.stack.enter_context(
            mock.patch.object(report, "verify_final_evaluation", side_effect=self.verify_final)
        )
        self.stack.enter_context(
            mock.patch.object(report, "verify_final_evaluation_gate", side_effect=self.verify_gate)
        )

    def _runs(self, task: str) -> list[dict[str, object]]:
        runs = []
        for seed in report.SEED_ORDER:
            run = self.project / "runs" / task / f"{task}_seed_{seed}"
            if task == "task1":
                history = [
                    {
                        "epoch": 1,
                        "elapsed_seconds": 3.25,
                        "train": {"clip_loss": 0.5},
                        "validation": {"clip_loss": 0.4, "accuracy": 0.7, "macro_f1": 0.68},
                        "checkpoint_improved": True,
                    },
                    {
                        "epoch": 2,
                        "elapsed_seconds": 3.0,
                        "train": {"clip_loss": 0.4},
                        "validation": {"clip_loss": 0.3, "accuracy": 0.75, "macro_f1": 0.72},
                        "checkpoint_improved": True,
                    },
                ]
            else:
                history = [
                    {
                        "epoch": 1,
                        "train": {"loss": 0.25},
                        "validation": {"loss": 0.22},
                        "checkpoint_improved": True,
                    },
                    {
                        "epoch": 2,
                        "train": {"loss": 0.2},
                        "validation": {"loss": 0.19},
                        "checkpoint_improved": True,
                    },
                ]
            history_record = _write_json(run / "epoch_history.json", history)
            result_record = _write_json(
                run / "result.json", {"artifacts": {"epoch_history": history_record}}
            )
            runs.append(
                {
                    "seed": seed,
                    "run_directory": str(run.resolve()),
                    "result": result_record,
                }
            )
        return runs

    def verify_final(self) -> dict[str, object]:
        self.final_calls += 1
        return self.final

    def verify_gate(self) -> dict[str, object]:
        self.gate_calls += 1
        return {
            "gate": self.gate,
            "gate_artifact": self.gate_record,
            "lock_artifact": self.gate_lock_record,
        }

    def close(self) -> None:
        self.stack.close()
        self.temporary.cleanup()


class FinalReportAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ReportFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_builds_and_recursively_verifies_fixed_evidence(self) -> None:
        built = report.build_final_report_assets()
        self.assertTrue(built["created"])
        self.assertEqual(built["manifest"]["asset_count"], len(report._ASSET_MEDIA_TYPES))
        self.assertEqual(
            set(path.name for path in self.fixture.report_root.iterdir()),
            report._COMPLETE_ENTRIES,
        )
        self.assertEqual(len(built["manifest"]["training_history_sources"]), 6)
        for name in report._ASSET_MEDIA_TYPES:
            payload = (self.fixture.report_root / name).read_bytes()
            self.assertTrue(payload)
            if name.endswith(".png"):
                self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        verified = report.verify_final_report_assets()
        self.assertFalse(verified["created"])
        self.assertEqual(verified["manifest"], built["manifest"])
        self.assertGreaterEqual(self.fixture.final_calls, 2)
        self.assertGreaterEqual(self.fixture.gate_calls, 2)

    def test_task2_interval_table_includes_threshold_and_pooled_counts(self) -> None:
        report.build_final_report_assets()
        path = self.fixture.report_root / "task2_seed37_metrics_intervals.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        pooled = next(
            row
            for row in rows
            if row["score_stream"] == "reconstruction"
            and row["scope"] == "pooled"
            and row["metric"] == "auroc"
        )
        self.assertEqual(float(pooled["threshold"]), 0.3)
        self.assertEqual(pooled["known_recording_count"], "267")
        self.assertEqual(pooled["unknown_recording_count"], "200")
        macro = next(
            row
            for row in rows
            if row["score_stream"] == "reconstruction"
            and row["scope"] == "macro"
            and row["metric"] == "auroc"
        )
        self.assertEqual(float(macro["threshold"]), 0.3)
        self.assertEqual(macro["known_recording_count"], "")
        self.assertEqual(macro["unknown_recording_count"], "")

    def test_verifier_does_not_create_a_missing_output_directory(self) -> None:
        self.assertFalse(self.fixture.report_root.exists())
        with self.assertRaises(OSError):
            report.verify_final_report_assets()
        self.assertFalse(self.fixture.report_root.exists())

    def test_second_build_is_idempotent_and_does_not_rewrite(self) -> None:
        first = report.build_final_report_assets()
        modified = {
            path.name: path.stat().st_mtime_ns for path in self.fixture.report_root.iterdir()
        }
        second = report.build_final_report_assets()
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(
            modified,
            {path.name: path.stat().st_mtime_ns for path in self.fixture.report_root.iterdir()},
        )

    def test_manifest_without_lock_is_recovered_without_asset_rewrite(self) -> None:
        report.build_final_report_assets()
        asset = self.fixture.report_root / "task1_seed_metrics.csv"
        modified = asset.stat().st_mtime_ns
        (self.fixture.report_root / "lock.json").unlink()
        recovered = report.build_final_report_assets()
        self.assertTrue(recovered["created"])
        self.assertEqual(asset.stat().st_mtime_ns, modified)
        self.assertTrue((self.fixture.report_root / "lock.json").is_file())

    def test_tampered_asset_is_rejected_even_when_manifest_and_lock_remain(self) -> None:
        report.build_final_report_assets()
        (self.fixture.report_root / "task1_seed_metrics.csv").write_text(
            "seed,accuracy,macro_f1\n13,0,0\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            report.verify_final_report_assets()

    def test_noncanonical_manifest_is_rejected_even_with_a_rebound_lock(self) -> None:
        built = report.build_final_report_assets()
        manifest_path = self.fixture.report_root / "manifest.json"
        compact = json.dumps(built["manifest"], sort_keys=True).encode("utf-8")
        manifest_path.write_bytes(compact)
        manifest_record = {
            "path": "manifest.json",
            "sha256": hashlib.sha256(compact).hexdigest(),
            "size_bytes": len(compact),
        }
        (self.fixture.report_root / "lock.json").write_bytes(
            _json_bytes(
                {
                    "schema_version": report.FINAL_REPORT_ASSETS_SCHEMA_VERSION,
                    "asset_set_id": report.FINAL_REPORT_ASSET_SET_ID,
                    "manifest": manifest_record,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            report.verify_final_report_assets()

    def test_extra_file_and_symlink_are_rejected(self) -> None:
        report.build_final_report_assets()
        extra = self.fixture.report_root / "extra.txt"
        extra.write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected entries"):
            report.verify_final_report_assets()
        extra.unlink()
        outside = self.fixture.project / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        (self.fixture.report_root / "extra.txt").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "unexpected entries"):
            report.verify_final_report_assets()

    def test_source_drift_is_rejected_before_asset_generation(self) -> None:
        with (
            mock.patch.object(report, "source_fingerprint", return_value="b" * 64),
            mock.patch.object(report, "_asset_payloads") as payloads,
            self.assertRaisesRegex(PermissionError, "source fingerprint"),
        ):
            report.build_final_report_assets()
        payloads.assert_not_called()

    def test_gate_bound_history_tamper_is_rejected(self) -> None:
        history_path = (
            self.fixture.project / "runs" / "task1" / "task1_seed_13" / "epoch_history.json"
        )
        history_path.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from its verified descriptor"):
            report.build_final_report_assets()

    def test_public_apis_accept_no_output_override(self) -> None:
        self.assertEqual(tuple(inspect.signature(report.build_final_report_assets).parameters), ())
        self.assertEqual(tuple(inspect.signature(report.verify_final_report_assets).parameters), ())
        self.assertNotIn("load_locked", report.__dict__)
        self.assertNotIn("run_final_evaluation", report.__dict__)


if __name__ == "__main__":
    unittest.main()
