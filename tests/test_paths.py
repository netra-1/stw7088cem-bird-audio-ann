from __future__ import annotations

import unittest

from bird_audio.paths import RAW_DATA_ROOT, require_safe_output


class RawDataSafetyTests(unittest.TestCase):
    def test_rejects_output_inside_raw_dataset(self) -> None:
        with self.assertRaises(ValueError):
            require_safe_output(RAW_DATA_ROOT / "derived.csv")

    def test_accepts_manifest_output(self) -> None:
        path = require_safe_output("data/manifests/example.csv")
        self.assertIn("data/manifests", path.as_posix())

    def test_rejects_output_over_source_or_outside_project(self) -> None:
        with self.assertRaises(ValueError):
            require_safe_output("src/bird_audio/generated.txt")
        with self.assertRaises(ValueError):
            require_safe_output("/tmp/generated.txt")


if __name__ == "__main__":
    unittest.main()
