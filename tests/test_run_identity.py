from __future__ import annotations

import unittest
from datetime import UTC, datetime

from bird_audio.run_identity import make_run_id


class RunIdentityTests(unittest.TestCase):
    def test_run_identifier_is_stable_and_sanitized(self) -> None:
        run_id = make_run_id(
            task="T1",
            rung="Final",
            seed=37,
            config_hash="a" * 64,
            data_hash="b" * 64,
            when=datetime(2026, 7, 13, 12, 30, tzinfo=UTC),
        )
        self.assertEqual(
            run_id,
            "20260713T123000Z_t1_final_s0037_caaaaaaaa_dbbbbbbbb",
        )


if __name__ == "__main__":
    unittest.main()
