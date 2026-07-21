from __future__ import annotations

import unittest

from bird_audio.paths import PROJECT_ROOT

TEXT_SUFFIXES = {".py", ".toml", ".md", ".csv", ".in", ".example"}
TOP_LEVEL_TEXT_FILES = {".gitignore", "README.md"}


class TextPolicyTests(unittest.TestCase):
    def test_project_text_has_no_en_or_em_dash(self) -> None:
        candidates = [
            path
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file()
            and ".venv" not in path.parts
            and "dataset" not in path.parts
            and "data" not in path.parts
            and (path.suffix in TEXT_SUFFIXES or path.name in TOP_LEVEL_TEXT_FILES)
        ]
        violations: list[str] = []
        for path in candidates:
            text = path.read_text(encoding="utf-8")
            if "\u2013" in text or "\u2014" in text:
                violations.append(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
