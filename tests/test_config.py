from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bird_audio.config import (
    ConfigValidationError,
    config_fingerprint,
    deep_merge,
    load_toml,
    public_config,
    validate_config,
    validate_project_config_set,
)
from bird_audio.paths import PROJECT_ROOT, resolve_project_path
from bird_audio.unknown_audio import load_unknown_audio_config


class ConfigTests(unittest.TestCase):
    def test_deep_merge_keeps_sibling_values(self) -> None:
        base = {"training": {"lr": 0.1, "epochs": 10}, "seed": 13}
        override = {"training": {"lr": 0.01}}
        self.assertEqual(
            deep_merge(base, override),
            {"training": {"lr": 0.01, "epochs": 10}, "seed": 13},
        )

    def test_final_task1_config_has_a_stable_fingerprint(self) -> None:
        config = load_toml("configs/task1/final.toml")
        self.assertEqual(config["rung"], "final")
        self.assertEqual(config["training"]["head_learning_rate"], 0.0003)
        self.assertEqual(config["training"]["maximum_epochs"], 30)
        self.assertEqual(config["seeds"], [13, 37, 71])
        self.assertEqual(config_fingerprint(config), config_fingerprint(config))
        self.assertNotIn("_config_path", public_config(config))

    def test_task1_method_choices_are_strictly_locked(self) -> None:
        base = public_config(load_toml("configs/task1/final.toml"))
        mutations = []

        wrong_loss = copy.deepcopy(base)
        wrong_loss["loss"]["name"] = "focal"
        mutations.append(wrong_loss)

        wrong_strategy = copy.deepcopy(base)
        wrong_strategy["sampling"]["strategy"] = "uniform"
        mutations.append(wrong_strategy)

        wrong_batch = copy.deepcopy(base)
        wrong_batch["training"]["batch_size"] = 16
        mutations.append(wrong_batch)

        disabled_augmentation = copy.deepcopy(base)
        disabled_augmentation["augmentation"]["specaugment"] = False
        mutations.append(disabled_augmentation)

        for config in mutations:
            with self.subTest(config=config), self.assertRaises(ConfigValidationError):
                validate_config(config)

    def test_every_task1_final_field_is_strictly_locked(self) -> None:
        base = public_config(load_toml("configs/task1/final.toml"))

        def leaf_paths(mapping, prefix=()):
            for key, value in mapping.items():
                path = (*prefix, key)
                if isinstance(value, dict):
                    yield from leaf_paths(value, path)
                else:
                    yield path, value

        for path, value in leaf_paths(base):
            mutated = copy.deepcopy(base)
            target = mutated
            for key in path[:-1]:
                target = target[key]
            if isinstance(value, bool):
                target[path[-1]] = not value
            elif isinstance(value, str):
                target[path[-1]] = f"{value}_changed"
            elif isinstance(value, int):
                target[path[-1]] = value + 1
            elif isinstance(value, float):
                target[path[-1]] = value + 0.125
            elif isinstance(value, list):
                target[path[-1]] = [*value, "changed"]
            else:
                self.fail(f"Unsupported Task 1 configuration value at {'.'.join(path)}")

            with self.subTest(path=".".join(path)), self.assertRaises(ConfigValidationError):
                validate_config(mutated)

    def test_every_task2_autoencoder_field_is_strictly_locked(self) -> None:
        base = public_config(load_toml("configs/task2/autoencoder.toml"))

        def leaf_paths(mapping, prefix=()):
            for key, value in mapping.items():
                path = (*prefix, key)
                if isinstance(value, dict):
                    yield from leaf_paths(value, path)
                else:
                    yield path, value

        leaves = tuple(leaf_paths(base))
        self.assertEqual(len(leaves), 56)
        for path, value in leaves:
            mutated = copy.deepcopy(base)
            target = mutated
            for key in path[:-1]:
                target = target[key]
            if isinstance(value, bool):
                target[path[-1]] = not value
            elif isinstance(value, str):
                target[path[-1]] = f"{value}_changed"
            elif isinstance(value, int):
                target[path[-1]] = value + 1
            elif isinstance(value, float):
                target[path[-1]] = value + 0.125
            elif isinstance(value, list):
                target[path[-1]] = [*value, "changed"]
            else:
                self.fail(f"Unsupported Task 2 configuration value at {'.'.join(path)}")

            with self.subTest(path=".".join(path)), self.assertRaises(ConfigValidationError):
                validate_config(mutated)

    def test_task2_numeric_types_are_strictly_locked(self) -> None:
        config = public_config(load_toml("configs/task2/autoencoder.toml"))
        config["training"]["batch_size"] = 64.0
        with self.assertRaises(ConfigValidationError):
            validate_config(config)

    def test_task1_class_order_matches_data_known_species_exactly(self) -> None:
        task1 = load_toml("configs/task1/final.toml")
        data = load_toml("configs/data.toml")
        expected = [entry["common_name"] for entry in data["known_species"]]
        self.assertEqual(task1["class_order"], expected)

    def test_rejects_unrecognized_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.toml"
            path.write_text("unexpected = true\n", encoding="utf-8")
            with self.assertRaises(ConfigValidationError):
                load_toml(path)

    def test_all_project_configs_are_semantically_consistent(self) -> None:
        counts = validate_project_config_set()
        self.assertEqual(counts["task1_configs"], 1)
        self.assertEqual(counts["task2_configs"], 1)
        self.assertEqual(counts["unknown_acquisition_configs"], 1)
        self.assertEqual(counts["unknown_selection_configs"], 1)
        self.assertEqual(counts["unknown_audio_configs"], 1)

    def test_task2_config_directory_contains_only_the_canonical_file(self) -> None:
        observed = sorted((PROJECT_ROOT / "configs" / "task2").glob("*.toml"))
        self.assertEqual(
            observed,
            [PROJECT_ROOT / "configs" / "task2" / "autoencoder.toml"],
        )

    def test_project_validation_rejects_noncanonical_task2_file_sets(self) -> None:
        source = (PROJECT_ROOT / "configs" / "task2" / "autoencoder.toml").read_text(
            encoding="utf-8"
        )
        for case in ("extra", "renamed"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                if case == "extra":
                    (directory / "autoencoder.toml").write_text(source, encoding="utf-8")
                    (directory / "alternative.toml").write_text(source, encoding="utf-8")
                else:
                    (directory / "renamed.toml").write_text(source, encoding="utf-8")

                def resolve_with_temporary_task2(path, task2_directory=directory):
                    if str(path) == "configs/task2":
                        return task2_directory.resolve()
                    if str(path) == "configs/task2/autoencoder.toml":
                        return (task2_directory / "autoencoder.toml").resolve()
                    return resolve_project_path(path)

                with (
                    patch(
                        "bird_audio.config.resolve_project_path",
                        side_effect=resolve_with_temporary_task2,
                    ),
                    self.assertRaisesRegex(ConfigValidationError, "Task 2 must contain exactly"),
                ):
                    validate_project_config_set()

    def test_unknown_selection_seed_must_match_data_split_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_path = Path(temporary) / "data.toml"
            source = (PROJECT_ROOT / "configs/data.toml").read_text(encoding="utf-8")
            data_path.write_text(
                source.replace("split_seed = 20260713", "split_seed = 20260714"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigValidationError, "selection seed"):
                validate_project_config_set(data_path=data_path)

    def test_unknown_selection_pool_target_must_match_acquisition(self) -> None:
        selection = {
            "selection_seed": 20260713,
            "candidate_pool_target_recordings_per_species": 81,
            "target_recordings_per_species": 40,
        }
        with (
            patch(
                "bird_audio.unknown_planning.load_unknown_selection_config",
                return_value=selection,
            ),
            self.assertRaisesRegex(ConfigValidationError, "candidate pool target"),
        ):
            validate_project_config_set()

    def test_unknown_species_protocol_is_strictly_locked(self) -> None:
        base = public_config(load_toml("configs/data.toml"))
        mutations = []

        zero_target = copy.deepcopy(base)
        zero_target["unknown_species"][0]["target_recordings"] = 0
        mutations.append(zero_target)

        duplicate_known = copy.deepcopy(base)
        duplicate_known["unknown_species"][0]["scientific_name"] = base["known_species"][0][
            "scientific_name"
        ]
        mutations.append(duplicate_known)

        invalid_group = copy.deepcopy(base)
        invalid_group["unknown_species"][0]["difficulty_group"] = "unlocked_group"
        mutations.append(invalid_group)

        missing_fallback = copy.deepcopy(base)
        missing_fallback["fallback_unknown_species"] = []
        mutations.append(missing_fallback)

        for config in mutations:
            with self.subTest(config=config), self.assertRaises(ConfigValidationError):
                validate_config(config)

    def test_unknown_audio_qc_must_match_data_protocol(self) -> None:
        unknown_audio = {
            "selection_seed": 20260713,
            "candidate_pool_target_recordings_per_species": 80,
            "target_recordings_per_species": 40,
            "inputs": {"known_manifest": "data/manifests/recordings.csv"},
            "audio_qc": {
                "minimum_source_sample_rate_hz": 44100,
                "full_decode_warning_policy": "exclude",
                "minimum_decoded_to_ffprobe_duration_ratio": 0.98,
                "maximum_decoded_to_ffprobe_duration_ratio": 1.02,
            },
            "session": {"coordinate_radius_km": 1.0},
        }
        with (
            patch(
                "bird_audio.unknown_audio.load_unknown_audio_config",
                return_value=unknown_audio,
            ),
            self.assertRaisesRegex(ConfigValidationError, "minimum source sample rate"),
        ):
            validate_project_config_set()

    def test_unknown_audio_candidate_plan_paths_are_locked(self) -> None:
        unknown_audio = copy.deepcopy(load_unknown_audio_config())
        unknown_audio["inputs"]["candidate_plan"] = "data/unknown/planning/unlocked.json"
        with (
            patch(
                "bird_audio.unknown_audio.load_unknown_audio_config",
                return_value=unknown_audio,
            ),
            self.assertRaisesRegex(ConfigValidationError, "candidate plan differs"),
        ):
            validate_project_config_set()

    def test_unknown_audio_quality_must_match_selection_strata(self) -> None:
        unknown_audio = copy.deepcopy(load_unknown_audio_config())
        unknown_audio["metadata"]["accepted_quality"] = ["A"]
        with (
            patch(
                "bird_audio.unknown_audio.load_unknown_audio_config",
                return_value=unknown_audio,
            ),
            self.assertRaisesRegex(ConfigValidationError, "accepted quality"),
        ):
            validate_project_config_set()


if __name__ == "__main__":
    unittest.main()
