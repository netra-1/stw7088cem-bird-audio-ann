from __future__ import annotations

import re
import unittest

from bird_audio import final_evaluation_data as final_data
from bird_audio import final_evaluation_gate as gate
from bird_audio import final_report_assets as report
from bird_audio import task1_attribution as attribution
from bird_audio import task1_training, task2_training, unknown_clip_cache
from bird_audio.paths import PROJECT_ROOT, is_relative_to
from bird_audio.provenance import (
    DEFAULT_ENVIRONMENT_V2_PATH,
    DEFAULT_MPS_SMOKE_CHECKPOINT_V2_PATH,
    DEFAULT_MPS_SMOKE_V2_PATH,
    DEFAULT_SIGNAL_SMOKE_V2_PATH,
    PROVENANCE_V2_ROOT,
)
from bird_audio.recovery_v2 import EQUIVALENCE_ROOT, RECOVERY_BUNDLE


class ReleaseV2NamespaceTests(unittest.TestCase):
    def test_active_release_uses_only_v2_output_namespaces(self) -> None:
        self.assertEqual(unknown_clip_cache.CACHE_VERSION, "unknown_clips_v2")
        self.assertEqual(
            unknown_clip_cache.DEFAULT_UNKNOWN_CLIP_CACHE_ROOT,
            "data/processed/unknown_clips_v2",
        )
        self.assertEqual(task1_training.DEFAULT_RUN_ROOT, PROJECT_ROOT / "runs" / "task1_v2")
        self.assertEqual(task2_training.DEFAULT_RUN_ROOT, PROJECT_ROOT / "runs" / "task2_v2")
        self.assertEqual(
            final_data.FINAL_EVALUATION_ROOT,
            PROJECT_ROOT / "runs" / "final_evaluation_v2",
        )
        self.assertEqual(gate.FINAL_EVALUATION_GATE_ID, "final_evaluation_gate_v2")
        self.assertEqual(
            gate.FINAL_EVALUATION_GATE_DIRECTORY,
            PROJECT_ROOT / "runs" / "final_evaluation_v2" / "gate_v2",
        )
        self.assertEqual(final_data.FINAL_EVALUATION_ATTEMPT_ID, "final_evaluation_attempt_v2")
        self.assertEqual(
            final_data.FINAL_EVALUATION_CLAIM_PATH,
            PROJECT_ROOT / "runs" / "final_evaluation_v2" / "final_evaluation_attempt_v2.json",
        )
        self.assertEqual(
            final_data.FINAL_EVALUATION_ATTEMPT_DIRECTORY,
            PROJECT_ROOT / "runs" / "final_evaluation_v2" / "attempt_v2",
        )
        self.assertEqual(report.FINAL_REPORT_ASSET_SET_ID, "final_report_assets_v2")
        self.assertEqual(
            report.FINAL_REPORT_ASSET_ROOT,
            PROJECT_ROOT / "report_assets" / "final_v2",
        )
        self.assertEqual(attribution.ATTRIBUTION_ID, "task1_attribution_v2")
        self.assertEqual(
            attribution.ATTRIBUTION_ROOT,
            PROJECT_ROOT / "report_assets" / "task1_attribution_v2",
        )
        self.assertEqual(
            EQUIVALENCE_ROOT,
            PROJECT_ROOT / "evidence" / "recovery" / "final_evaluation_v2_release_v1",
        )

    def test_active_bindings_have_one_unknown_cache_identity(self) -> None:
        expected_root = PROJECT_ROOT / "data" / "processed" / "unknown_clips_v2"
        expected_lock = "222ca630ce28ea05998c74592ad6c47795cde75176db1fcce6930dcbf49fe91b"
        self.assertEqual(gate.UNKNOWN_CACHE_LOCK_PATH.parent, expected_root)
        self.assertEqual(gate.EXPECTED_UNKNOWN_CACHE_LOCK_SHA256, expected_lock)
        self.assertEqual(final_data.UNKNOWN_CACHE_ROOT, expected_root)
        self.assertEqual(
            final_data.UNKNOWN_CACHE_LOCK_SHA256,
            gate.EXPECTED_UNKNOWN_CACHE_LOCK_SHA256,
        )
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", gate.EXPECTED_UNKNOWN_CACHE_LOCK_SHA256))

    def test_v2_outputs_are_outside_every_preserved_v1_tree(self) -> None:
        preserved_roots = (
            PROJECT_ROOT / "runs" / "task1",
            PROJECT_ROOT / "runs" / "task2",
            PROJECT_ROOT / "runs" / "final_evaluation",
            PROJECT_ROOT / "report_assets" / "provenance",
            PROJECT_ROOT / "report_assets" / "final_v1",
            PROJECT_ROOT / "report_assets" / "task1_attribution_v1",
            PROJECT_ROOT / "data" / "processed" / "unknown_clips_v1",
            RECOVERY_BUNDLE,
        )
        active_roots = (
            task1_training.DEFAULT_RUN_ROOT,
            task2_training.DEFAULT_RUN_ROOT,
            final_data.FINAL_EVALUATION_ROOT,
            PROVENANCE_V2_ROOT,
            final_data.UNKNOWN_CACHE_ROOT,
            report.FINAL_REPORT_ASSET_ROOT,
            attribution.ATTRIBUTION_ROOT,
            EQUIVALENCE_ROOT,
        )
        for active in active_roots:
            for preserved in preserved_roots:
                with self.subTest(active=active, preserved=preserved):
                    self.assertFalse(is_relative_to(active, preserved))
                    self.assertFalse(is_relative_to(preserved, active))

    def test_v2_runtime_evidence_shares_one_provenance_root(self) -> None:
        paths = (
            DEFAULT_ENVIRONMENT_V2_PATH,
            DEFAULT_MPS_SMOKE_V2_PATH,
            DEFAULT_MPS_SMOKE_CHECKPOINT_V2_PATH,
            DEFAULT_SIGNAL_SMOKE_V2_PATH,
            task1_training.DEFAULT_BENCHMARK_RESULT_PATH,
            task1_training.DEFAULT_BENCHMARK_LOCK_PATH,
            task2_training.DEFAULT_BENCHMARK_RESULT_PATH,
            task2_training.DEFAULT_BENCHMARK_LOCK_PATH,
        )
        self.assertTrue(all(path.parent == PROVENANCE_V2_ROOT for path in paths))


if __name__ == "__main__":
    unittest.main()
