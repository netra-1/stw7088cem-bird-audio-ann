from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from bird_audio.hashing import sha256_file, sha256_json
from bird_audio.paths import PROJECT_ROOT
from bird_audio.unknown_acquisition import LOCKED_UNKNOWN_SPECIES
from bird_audio.unknown_planning import (
    ASSIGNMENT_CANDIDATE_FIELDS,
    ASSIGNMENT_SLOT_FIELDS,
    SELECTION_SEED,
    UnknownPlanningError,
    allocate_reference_slots,
    assign_candidates_to_slots,
    build_candidate_queues,
    build_unknown_candidate_plan,
    load_unknown_selection_config,
    validate_unknown_candidate_plan,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "unknown_selection.toml"


def _recording(scientific_name: str, identifier: int) -> dict[str, object]:
    genus, specific_epithet = scientific_name.split()
    return {
        "id": str(identifier),
        "nr": str(identifier),
        "gen": genus,
        "sp": specific_epithet,
        "grp": "birds",
        "en": "Test bird",
        "q": "A",
        "length": "12.0",
        "lic": "https://creativecommons.org/licenses/by/4.0/",
        "url": f"https://xeno-canto.org/{identifier}",
        "file": f"https://xeno-canto.org/{identifier}/download",
    }


def _sealed_metadata(*, inventory_count: int = 80, reverse: bool = False) -> dict[str, object]:
    species: dict[str, object] = {}
    for species_index, identity in enumerate(LOCKED_UNKNOWN_SPECIES, start=1):
        role, active, common_name, scientific_name, difficulty_group = identity
        recordings = [
            _recording(scientific_name, species_index * 100_000 + offset)
            for offset in range(1, inventory_count + 1)
        ]
        page_boundary = inventory_count // 2
        left = recordings[:page_boundary]
        right = recordings[page_boundary:]
        if reverse:
            left.reverse()
            right.reverse()
            pages = {
                "2": {"recordings": right},
                "1": {"recordings": left},
            }
        else:
            pages = {
                "1": {"recordings": left},
                "2": {"recordings": right},
            }
        species[scientific_name] = {
            "role": role,
            "active": active,
            "common_name": common_name,
            "scientific_name": scientific_name,
            "difficulty_group": difficulty_group,
            "snapshot": {"num_recordings": inventory_count},
            "pages": pages,
        }
    return {"species": species}


def _descriptor(
    index: int,
    *,
    container: str = "mp3",
    duration: str = "12.0",
) -> dict[str, object]:
    if float(duration) < 3:
        duration_bucket = "below_3"
    elif float(duration) < 10:
        duration_bucket = "3_to_below_10"
    elif float(duration) < 30:
        duration_bucket = "10_to_below_30"
    elif float(duration) < 60:
        duration_bucket = "30_to_below_60"
    else:
        duration_bucket = "at_least_60"
    return {
        "recording_id": f"XC{index}",
        "sha256": sha256_json({"recording": index}),
        "session_group": f"session:{index}",
        "container": container,
        "source_rate_bucket": "48000",
        "channels": "stereo",
        "quality": "A",
        "duration_bucket": duration_bucket,
        "duration_seconds": duration,
    }


def _slot(index: int, *, container: str = "mp3", duration: str = "12.0") -> dict[str, str]:
    source = _descriptor(index, container=container, duration=duration)
    return {
        "slot_id": f"slot-{index:02d}",
        **{field: str(source[field]) for field in ASSIGNMENT_SLOT_FIELDS if field != "slot_id"},
    }


def _candidate(index: int, *, container: str = "mp3", duration: str = "12.0") -> dict[str, str]:
    source = _descriptor(index, container=container, duration=duration)
    return {
        "candidate_id": f"candidate-{index:02d}",
        "session_group": f"unknown-session:{index}",
        **{
            field: str(source[field])
            for field in ASSIGNMENT_CANDIDATE_FIELDS
            if field not in {"candidate_id", "session_group"}
        },
    }


class UnknownPlanningPureTests(unittest.TestCase):
    def test_config_is_strict_and_locked(self) -> None:
        config = load_unknown_selection_config(CONFIG_PATH)
        self.assertEqual(config["selection_seed"], SELECTION_SEED)
        self.assertEqual(config["target_recordings_per_species"], 40)
        self.assertEqual(config["candidate_pool_target_recordings_per_species"], 80)

    def test_candidate_queue_inventory_boundaries_and_order_independence(self) -> None:
        for inventory_count in (39, 40, 79, 80, 81):
            with self.subTest(inventory_count=inventory_count):
                forward = build_candidate_queues(_sealed_metadata(inventory_count=inventory_count))
                reverse = build_candidate_queues(
                    _sealed_metadata(inventory_count=inventory_count, reverse=True)
                )
                self.assertEqual(forward, reverse)
                expected_shortfall = max(0, 80 - inventory_count)
                expected_status = (
                    "complete_inventory_below_target"
                    if expected_shortfall
                    else "inventory_at_or_above_target"
                )
                for queue in forward:
                    self.assertEqual(queue["inventory_recordings"], inventory_count)
                    self.assertEqual(len(queue["candidates"]), inventory_count)
                    self.assertEqual(queue["candidate_pool_target_recordings"], 80)
                    self.assertEqual(queue["candidate_pool_inventory_status"], expected_status)
                    self.assertEqual(
                        queue["candidate_pool_inventory_shortfall_recordings"],
                        expected_shortfall,
                    )
                    self.assertEqual(queue["target_recordings"], 40)

                primary_queues = [queue for queue in forward if queue["role"] == "primary"]
                self.assertTrue(
                    all(
                        queue["activation_status"] == "active_primary_queue"
                        for queue in primary_queues
                    )
                )
                fallback = [queue for queue in forward if queue["role"] == "fallback"]
                self.assertEqual(
                    fallback[0]["activation_status"], "inactive_fallback_until_protocol_gate"
                )

    def test_fallback_queue_is_present_but_inactive(self) -> None:
        queues = build_candidate_queues(_sealed_metadata())
        fallback = [queue for queue in queues if queue["role"] == "fallback"]
        self.assertEqual(len(fallback), 1)
        self.assertFalse(fallback[0]["active"])
        self.assertEqual(
            fallback[0]["activation_status"],
            "inactive_fallback_until_protocol_gate",
        )

    def test_candidate_queue_rejects_unapproved_outcome_field(self) -> None:
        metadata = _sealed_metadata()
        species = next(iter(metadata["species"].values()))
        species["pages"]["1"]["recordings"][0]["model_score"] = 0.99
        with self.assertRaisesRegex(UnknownPlanningError, "unapproved fields"):
            build_candidate_queues(metadata)

    def test_largest_remainder_produces_exactly_40_slots(self) -> None:
        descriptors = [
            _descriptor(index, container="mp3" if index <= 70 else "riff_wave")
            for index in range(1, 101)
        ]
        reference = allocate_reference_slots(descriptors)
        self.assertEqual(reference["target_slots"], 40)
        self.assertEqual(len(reference["reference_slots"]), 40)
        allocations = {
            item["stratum"]["container"]: item["allocated_slots"]
            for item in reference["allocations"]
        }
        self.assertEqual(allocations, {"mp3": 28, "riff_wave": 12})

    def test_hungarian_ties_are_stable_under_input_reordering(self) -> None:
        slots = [_slot(index) for index in range(1, 41)]
        candidates = [_candidate(index) for index in range(1, 41)]
        forward = assign_candidates_to_slots(slots, candidates)
        reverse = assign_candidates_to_slots(list(reversed(slots)), list(reversed(candidates)))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["total_categorical_mismatches"], 0)
        self.assertEqual(forward["total_duration_distance_units"], 0)

    def test_one_categorical_mismatch_cannot_be_bought_by_duration(self) -> None:
        slots = [_slot(index, duration="10.0") for index in range(1, 41)]
        exact_category = [
            _candidate(index, container="mp3", duration="29.999999") for index in range(1, 41)
        ]
        exact_duration = [
            _candidate(index + 40, container="riff_wave", duration="10.0") for index in range(1, 41)
        ]
        result = assign_candidates_to_slots(slots, [*exact_category, *exact_duration])
        selected = {item["candidate_id"] for item in result["assignments"]}
        self.assertEqual(selected, {item["candidate_id"] for item in exact_category})
        self.assertEqual(result["total_categorical_mismatches"], 0)

    def test_assignment_rejects_any_extra_outcome_field(self) -> None:
        slots = [_slot(index) for index in range(1, 41)]
        candidates = [_candidate(index) for index in range(1, 41)]
        candidates[0]["test_outcome"] = "positive"
        with self.assertRaisesRegex(UnknownPlanningError, "fields are invalid"):
            assign_candidates_to_slots(slots, candidates)


class UnknownPlanningArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        unknown_root = PROJECT_ROOT / "data" / "unknown"
        unknown_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="planning-test-", dir=unknown_root)
        self.root = Path(self.temporary.name)
        self.config = self.root / "unknown_selection.toml"
        self.config.write_bytes(CONFIG_PATH.read_bytes())
        self.metadata = self.root / "unknown_metadata.json"
        self.metadata.write_text(
            json.dumps(_sealed_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.metadata_lock = self.root / "unknown_metadata_lock.json"
        self.metadata_lock.write_text("{}\n", encoding="utf-8")
        self.manifest = self.root / "recordings.csv"
        self.review_lock = self.root / "review_lock.json"
        self.review_lock.write_text("{}\n", encoding="utf-8")
        self.split = self.root / "split.csv"
        self.summary = self.root / "split_summary.json"
        self.split_lock = self.root / "split_lock.json"
        self.plan = self.root / "candidate_plan.json"
        self.plan_lock = self.root / "candidate_plan_lock.json"
        self._write_known_artifacts()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_known_artifacts(self) -> None:
        manifest_rows: list[dict[str, str]] = []
        split_rows: list[dict[str, str]] = []
        for index in range(1, 51):
            recording_id = f"XC{900000 + index}"
            digest = sha256_json({"known": index})
            session = f"session:known-{index}"
            common = "Known test bird"
            relative = f"dataset/Test/{900000 + index}.mp3"
            manifest_rows.append(
                {
                    "recording_id": recording_id,
                    "relative_path": relative,
                    "sha256": digest,
                    "species_common_name": common,
                    "session_group": session,
                    "local_qc_status": "include",
                    "header_type": "mp3_id3" if index <= 30 else "riff_wave",
                    "source_sample_rate_hz": "48000",
                    "channels": "2",
                    "quality": "A",
                    "canonical_duration_seconds": str(2 + index),
                }
            )
        self._write_csv(self.manifest, list(manifest_rows[0]), manifest_rows)
        manifest_sha256 = sha256_file(self.manifest)
        for row in manifest_rows:
            split_rows.append(
                {
                    "recording_id": row["recording_id"],
                    "relative_path": row["relative_path"],
                    "sha256": row["sha256"],
                    "species_common_name": row["species_common_name"],
                    "session_group": row["session_group"],
                    "split": "test",
                    "split_seed": str(SELECTION_SEED),
                    "source_manifest_sha256": manifest_sha256,
                }
            )
        self._write_csv(self.split, list(split_rows[0]), split_rows)
        split_sha256 = sha256_file(self.split)
        review_sha256 = sha256_file(self.review_lock)
        summary = {
            "schema_version": "1.2",
            "source_manifest_sha256": manifest_sha256,
            "review_lock_sha256": review_sha256,
            "split_sha256": split_sha256,
            "split_seed": SELECTION_SEED,
            "recordings": len(split_rows),
        }
        self.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lock = {
            "schema_version": "1.2",
            "source_manifest_sha256": manifest_sha256,
            "review_lock_sha256": review_sha256,
            "split_sha256": split_sha256,
            "summary_sha256": sha256_file(self.summary),
            "split_seed": SELECTION_SEED,
            "recordings": len(split_rows),
            "recording_set_sha256": sha256_json(sorted(row["recording_id"] for row in split_rows)),
        }
        self.split_lock.write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @contextmanager
    def _verified_sources(self):
        with ExitStack() as stack:
            unknown = stack.enter_context(
                patch("bird_audio.unknown_planning.verify_unknown_metadata_lock")
            )
            review = stack.enter_context(patch("bird_audio.unknown_planning.verify_review_lock"))
            unknown.side_effect = lambda *_args, **_kwargs: {
                "ready_for_candidate_planning": True,
                "sealed_cache_sha256": sha256_file(self.metadata),
                "candidate_pool_target_recordings_per_species": 80,
                "target_recordings_per_species": 40,
                "primary_species_count": 5,
                "inactive_fallback_count": 1,
            }
            review.side_effect = lambda *_args, **_kwargs: {
                "ready_for_split": True,
                "final_manifest_sha256": sha256_file(self.manifest),
            }
            yield

    def _build(self) -> None:
        build_unknown_candidate_plan(
            self.config,
            self.metadata,
            self.metadata_lock,
            self.manifest,
            self.review_lock,
            self.split,
            self.summary,
            self.split_lock,
            self.plan,
            self.plan_lock,
        )

    def test_plan_is_create_only_and_idempotent_without_overwrite(self) -> None:
        with self._verified_sources():
            self._build()
            first_plan = sha256_file(self.plan)
            first_lock = sha256_file(self.plan_lock)
            self._build()
        self.assertEqual(sha256_file(self.plan), first_plan)
        self.assertEqual(sha256_file(self.plan_lock), first_lock)

    def test_plan_accepts_complete_inventory_below_candidate_pool_target(self) -> None:
        self.metadata.write_text(
            json.dumps(_sealed_metadata(inventory_count=79), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self._verified_sources():
            self._build()
            result = validate_unknown_candidate_plan(self.plan_lock, self.plan)

        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        self.assertEqual(result["candidate_recordings_total"], 6 * 79)
        for queue in plan["candidate_queues"]:
            self.assertEqual(queue["inventory_recordings"], 79)
            self.assertEqual(
                queue["candidate_pool_inventory_status"], "complete_inventory_below_target"
            )
            self.assertEqual(queue["candidate_pool_inventory_shortfall_recordings"], 1)
            self.assertEqual(queue["target_recordings"], 40)

    def test_existing_plan_refuses_input_path_rebinding(self) -> None:
        alternate_config = self.root / "alternate_unknown_selection.toml"
        alternate_config.write_bytes(self.config.read_bytes())
        with self._verified_sources():
            self._build()
            with self.assertRaisesRegex(UnknownPlanningError, "cannot be rebound"):
                build_unknown_candidate_plan(
                    alternate_config,
                    self.metadata,
                    self.metadata_lock,
                    self.manifest,
                    self.review_lock,
                    self.split,
                    self.summary,
                    self.split_lock,
                    self.plan,
                    self.plan_lock,
                )

    def test_validation_rejects_selection_config_drift(self) -> None:
        with self._verified_sources():
            self._build()
            self.config.write_text(
                self.config.read_text(encoding="utf-8").replace(
                    "selection_seed = 20260713", "selection_seed = 20260714"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(UnknownPlanningError, "artifact hash mismatch"):
                validate_unknown_candidate_plan(self.plan_lock, self.plan)

    def test_validation_rejects_split_drift(self) -> None:
        with self._verified_sources():
            self._build()
            self.split.write_text(self.split.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(UnknownPlanningError, "artifact hash mismatch"):
                validate_unknown_candidate_plan(self.plan_lock, self.plan)

    def test_validation_rejects_sealed_metadata_drift(self) -> None:
        with self._verified_sources():
            self._build()
            metadata = json.loads(self.metadata.read_text(encoding="utf-8"))
            metadata["species"]["Psilopogon zeylanicus"]["pages"]["1"]["recordings"][0]["q"] = "B"
            self.metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(UnknownPlanningError, "artifact hash mismatch"):
                validate_unknown_candidate_plan(self.plan_lock, self.plan)

    def test_validation_rejects_invalid_lock_timestamp(self) -> None:
        with self._verified_sources():
            self._build()
            lock = json.loads(self.plan_lock.read_text(encoding="utf-8"))
            lock["locked_at_utc"] = "not-a-timestamp"
            self.plan_lock.write_text(
                json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(UnknownPlanningError, "valid UTC timestamp"):
                validate_unknown_candidate_plan(self.plan_lock, self.plan)

    def test_planning_never_calls_network_or_audio_subprocess(self) -> None:
        with (
            self._verified_sources(),
            patch.object(urllib.request, "urlopen", side_effect=AssertionError("network call")),
            patch.object(subprocess, "run", side_effect=AssertionError("audio subprocess")),
        ):
            self._build()
        self.assertTrue(self.plan.is_file())
        self.assertTrue(self.plan_lock.is_file())


if __name__ == "__main__":
    unittest.main()
