from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bird_audio import recovery_v2 as recovery
from bird_audio.paths import PROJECT_ROOT


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecoveryManifestTests(unittest.TestCase):
    def test_pinned_v1_recovery_manifest_and_protected_trees_verify(self) -> None:
        result = recovery.verify_v1_recovery_manifest()

        self.assertTrue(result["valid"])
        self.assertEqual(result["manifest_id"], recovery.RECOVERY_MANIFEST_ID)
        self.assertEqual(
            result["v1_source_fingerprint_sha256"],
            recovery.V1_SOURCE_FINGERPRINT_SHA256,
        )
        self.assertEqual(len(result["protected_trees"]), 8)
        self.assertFalse(result["failure_boundary"]["result_present"])
        self.assertEqual(result["failure_boundary"]["published_prediction_artifacts"], 0)

    def test_wrong_pinned_manifest_hash_is_rejected(self) -> None:
        with (
            mock.patch.object(recovery, "RECOVERY_MANIFEST_SHA256", "0" * 64),
            self.assertRaisesRegex(ValueError, "pinned SHA-256"),
        ):
            recovery.verify_v1_recovery_manifest()

    def test_historical_source_fingerprint_is_reconstructed_from_records(self) -> None:
        with (
            mock.patch.object(
                recovery,
                "_source_fingerprint_from_records",
                return_value="0" * 64,
            ),
            self.assertRaisesRegex(ValueError, "source snapshot identity"),
        ):
            recovery.verify_v1_recovery_manifest()


class UnknownCacheEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="recovery-v2-test-",
            dir=PROJECT_ROOT / "data" / "processed",
        )
        self.root = Path(self.temporary.name)
        self.v1 = self.root / "unknown_clips_v1"
        self.v2 = self.root / "unknown_clips_v2"
        self.feature_payload = b"identical-scientific-feature-bytes"
        self.feature_set_sha256 = "1" * 64
        self.v1_content_sha256 = "2" * 64
        self._build_cache(self.v1, "unknown_clips_v1", "3" * 64, self.v1_content_sha256)
        self._build_cache(self.v2, "unknown_clips_v2", "4" * 64, "5" * 64)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build_cache(
        self,
        root: Path,
        version: str,
        implementation_sha256: str,
        content_sha256: str,
    ) -> None:
        feature = root / "scoring" / "features" / "XC1.npy"
        feature.parent.mkdir(parents=True)
        feature.write_bytes(self.feature_payload)
        feature.chmod(0o600)
        index = root / "scoring" / "index.csv"
        index.write_bytes(
            b"clip_id,candidate_id,feature_file\nXC1_0,XC1,scoring/features/XC1.npy\n"
        )
        index.chmod(0o600)
        summary = {
            "cache_version": version,
            "totals": {
                "recordings": 1,
                "clips": 1,
                "feature_files": 1,
                "feature_bytes": len(self.feature_payload),
            },
        }
        lock = {
            "schema_version": "1.0",
            "cache_version": version,
            "provenance": {
                "implementation_sha256": implementation_sha256,
                "shared_input_sha256": "6" * 64,
            },
            "artifacts": {
                "features": {"feature_set_sha256": self.feature_set_sha256},
            },
            "cache_content_sha256": content_sha256,
        }
        write_json(root / "summary.json", summary)
        write_json(root / "lock.json", lock)

    def _patch_contract(self):
        return mock.patch.multiple(
            recovery,
            V1_UNKNOWN_CACHE_ROOT=self.v1,
            V2_UNKNOWN_CACHE_ROOT=self.v2,
            V1_UNKNOWN_CACHE_LOCK_SHA256=sha256_file(self.v1 / "lock.json"),
            V1_UNKNOWN_CACHE_CONTENT_SHA256=self.v1_content_sha256,
            V1_UNKNOWN_INDEX_SHA256=sha256_file(self.v1 / "scoring" / "index.csv"),
            V1_UNKNOWN_FEATURE_SET_SHA256=self.feature_set_sha256,
            V1_UNKNOWN_RECORDINGS=1,
            V1_UNKNOWN_CLIPS=1,
            V1_UNKNOWN_FEATURE_BYTES=len(self.feature_payload),
        )

    @staticmethod
    def _equivalence_result(*, ffmpeg=None, full_rederivation=True):
        _ = ffmpeg
        return {
            "valid": True,
            "full_rederivation": full_rederivation,
            "scientific_artifacts_identical": True,
        }

    @staticmethod
    def _certificate_value(*, full_rederivation: bool = True) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "equivalence_id": recovery.EQUIVALENCE_ID,
            "certified_at_utc": "2026-07-15T01:02:03+00:00",
            "source_fingerprint_sha256": "9" * 64,
            "complete": True,
            "equivalence": UnknownCacheEquivalenceTests._equivalence_result(
                full_rederivation=full_rederivation
            ),
        }

    def test_scientific_cache_artifacts_must_be_byte_identical(self) -> None:
        recovered = {
            "manifest": {"sha256": "7" * 64},
            "lock": {"sha256": "8" * 64},
        }
        verified = {
            "valid": True,
            "recordings": 1,
            "clips": 1,
            "feature_files": 1,
        }
        with (
            self._patch_contract(),
            mock.patch.object(recovery, "verify_v1_recovery_manifest", return_value=recovered),
            mock.patch.object(recovery, "verify_unknown_clip_cache", return_value=verified),
        ):
            result = recovery.verify_unknown_cache_v2_equivalence()

        self.assertTrue(result["scientific_artifacts_identical"])
        self.assertTrue(result["file_inodes_disjoint"])
        self.assertEqual(result["recordings"], 1)
        self.assertEqual(result["clips"], 1)

    def test_changed_v2_feature_is_rejected(self) -> None:
        feature = self.v2 / "scoring" / "features" / "XC1.npy"
        feature.write_bytes(b"changed")
        feature.chmod(0o600)
        recovered = {
            "manifest": {"sha256": "7" * 64},
            "lock": {"sha256": "8" * 64},
        }
        verified = {
            "valid": True,
            "recordings": 1,
            "clips": 1,
            "feature_files": 1,
        }
        with (
            self._patch_contract(),
            mock.patch.object(recovery, "verify_v1_recovery_manifest", return_value=recovered),
            mock.patch.object(recovery, "verify_unknown_clip_cache", return_value=verified),
            self.assertRaisesRegex(ValueError, "feature files differ"),
        ):
            recovery.verify_unknown_cache_v2_equivalence()

    def test_shared_v1_v2_index_inode_is_rejected(self) -> None:
        v1_index = self.v1 / "scoring" / "index.csv"
        v2_index = self.v2 / "scoring" / "index.csv"
        v2_index.unlink()
        os.link(v1_index, v2_index)
        recovered = {
            "manifest": {"sha256": "7" * 64},
            "lock": {"sha256": "8" * 64},
        }
        verified = {
            "valid": True,
            "recordings": 1,
            "clips": 1,
            "feature_files": 1,
        }
        with (
            self._patch_contract(),
            mock.patch.object(recovery, "verify_v1_recovery_manifest", return_value=recovered),
            mock.patch.object(recovery, "verify_unknown_clip_cache", return_value=verified),
            self.assertRaisesRegex(ValueError, "share the index file inode"),
        ):
            recovery.verify_unknown_cache_v2_equivalence()

    def test_equivalence_certificate_is_create_only_and_reverifiable(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="equivalence-certificate-test-",
            dir=PROJECT_ROOT / "evidence",
        ) as temporary:
            root = Path(temporary) / "certificate"
            path = root / "unknown_cache_equivalence.json"
            lock = root / "lock.json"

            with (
                mock.patch.multiple(
                    recovery,
                    EQUIVALENCE_ROOT=root,
                    EQUIVALENCE_PATH=path,
                    EQUIVALENCE_LOCK_PATH=lock,
                ),
                mock.patch.object(recovery, "source_fingerprint", return_value="9" * 64),
                mock.patch.object(
                    recovery,
                    "verify_unknown_cache_v2_equivalence",
                    side_effect=self._equivalence_result,
                ),
            ):
                created = recovery.seal_unknown_cache_v2_equivalence()
                verified = recovery.verify_unknown_cache_v2_equivalence_certificate()
                resealed = recovery.seal_unknown_cache_v2_equivalence()

            self.assertTrue(created["created"])
            self.assertFalse(verified["created"])
            self.assertFalse(resealed["created"])
            self.assertEqual({item.name for item in root.iterdir()}, {path.name, lock.name})
            self.assertEqual(os.stat(path).st_nlink, 1)

    def test_certificate_without_full_rederivation_is_rejected_before_light_check(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="equivalence-full-derivation-test-",
            dir=PROJECT_ROOT / "evidence",
        ) as temporary:
            root = Path(temporary) / "certificate"
            path = root / "unknown_cache_equivalence.json"
            lock = root / "lock.json"
            with mock.patch.multiple(
                recovery,
                EQUIVALENCE_ROOT=root,
                EQUIVALENCE_PATH=path,
                EQUIVALENCE_LOCK_PATH=lock,
            ):
                recovery._publish_equivalence_bundle(
                    self._certificate_value(full_rederivation=False)
                )
                with (
                    mock.patch.object(recovery, "source_fingerprint", return_value="9" * 64),
                    mock.patch.object(
                        recovery,
                        "verify_unknown_cache_v2_equivalence",
                    ) as verifier,
                    self.assertRaisesRegex(ValueError, "lacks full rederivation"),
                ):
                    recovery.verify_unknown_cache_v2_equivalence_certificate()

            verifier.assert_not_called()

    def test_certificate_requires_exact_top_level_schema(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="equivalence-schema-test-",
            dir=PROJECT_ROOT / "evidence",
        ) as temporary:
            root = Path(temporary) / "certificate"
            path = root / "unknown_cache_equivalence.json"
            lock = root / "lock.json"
            value = self._certificate_value()
            value["unexpected"] = True
            with mock.patch.multiple(
                recovery,
                EQUIVALENCE_ROOT=root,
                EQUIVALENCE_PATH=path,
                EQUIVALENCE_LOCK_PATH=lock,
            ):
                recovery._publish_equivalence_bundle(value)
                with (
                    mock.patch.object(recovery, "source_fingerprint", return_value="9" * 64),
                    mock.patch.object(
                        recovery,
                        "verify_unknown_cache_v2_equivalence",
                    ) as verifier,
                    self.assertRaisesRegex(ValueError, "certificate identity"),
                ):
                    recovery.verify_unknown_cache_v2_equivalence_certificate()

            verifier.assert_not_called()

    def test_certificate_requires_canonical_utc_timestamp(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="equivalence-timestamp-test-",
            dir=PROJECT_ROOT / "evidence",
        ) as temporary:
            root = Path(temporary) / "certificate"
            path = root / "unknown_cache_equivalence.json"
            lock = root / "lock.json"
            value = self._certificate_value()
            value["certified_at_utc"] = "2026-07-15T01:02:03"
            with mock.patch.multiple(
                recovery,
                EQUIVALENCE_ROOT=root,
                EQUIVALENCE_PATH=path,
                EQUIVALENCE_LOCK_PATH=lock,
            ):
                recovery._publish_equivalence_bundle(value)
                with (
                    mock.patch.object(recovery, "source_fingerprint", return_value="9" * 64),
                    mock.patch.object(
                        recovery,
                        "verify_unknown_cache_v2_equivalence",
                    ) as verifier,
                    self.assertRaisesRegex(ValueError, "certificate identity"),
                ):
                    recovery.verify_unknown_cache_v2_equivalence_certificate()

            verifier.assert_not_called()

    def test_source_change_during_full_rederivation_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="equivalence-source-change-test-",
            dir=PROJECT_ROOT / "evidence",
        ) as temporary:
            root = Path(temporary) / "certificate"
            path = root / "unknown_cache_equivalence.json"
            lock = root / "lock.json"
            with (
                mock.patch.multiple(
                    recovery,
                    EQUIVALENCE_ROOT=root,
                    EQUIVALENCE_PATH=path,
                    EQUIVALENCE_LOCK_PATH=lock,
                ),
                mock.patch.object(
                    recovery,
                    "source_fingerprint",
                    side_effect=["9" * 64, "8" * 64],
                ),
                mock.patch.object(
                    recovery,
                    "verify_unknown_cache_v2_equivalence",
                    side_effect=self._equivalence_result,
                ) as verifier,
                self.assertRaisesRegex(RuntimeError, "Source changed during full"),
            ):
                recovery.seal_unknown_cache_v2_equivalence()

            self.assertFalse(root.exists())
            self.assertTrue(verifier.call_args.kwargs["full_rederivation"])

    def test_publication_race_verifies_existing_certificate(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="equivalence-race-test-",
            dir=PROJECT_ROOT / "evidence",
        ) as temporary:
            root = Path(temporary) / "certificate"
            path = root / "unknown_cache_equivalence.json"
            lock = root / "lock.json"
            original_publish = recovery._publish_equivalence_bundle

            def publish_race(value):
                original_publish(value)
                raise FileExistsError("Concurrent certificate publication")

            with (
                mock.patch.multiple(
                    recovery,
                    EQUIVALENCE_ROOT=root,
                    EQUIVALENCE_PATH=path,
                    EQUIVALENCE_LOCK_PATH=lock,
                ),
                mock.patch.object(recovery, "source_fingerprint", return_value="9" * 64),
                mock.patch.object(
                    recovery,
                    "verify_unknown_cache_v2_equivalence",
                    side_effect=self._equivalence_result,
                ) as verifier,
                mock.patch.object(
                    recovery,
                    "_publish_equivalence_bundle",
                    side_effect=publish_race,
                ),
            ):
                result = recovery.seal_unknown_cache_v2_equivalence()

            self.assertFalse(result["created"])
            self.assertEqual(
                [call.kwargs["full_rederivation"] for call in verifier.call_args_list],
                [True, False],
            )


if __name__ == "__main__":
    unittest.main()
