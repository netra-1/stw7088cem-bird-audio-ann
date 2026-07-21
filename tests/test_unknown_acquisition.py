from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

import bird_audio.unknown_acquisition as unknown_acquisition
from bird_audio.metadata import XenoCantoApiError, XenoCantoFatalApiError
from bird_audio.paths import PROJECT_ROOT
from bird_audio.unknown_acquisition import (
    CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES,
    DEFAULT_ENDPOINT,
    LOCKED_UNKNOWN_SPECIES,
    SPECIES_QUERY_FORM,
    TARGET_RECORDINGS_PER_SPECIES,
    UnknownAcquisitionConfigError,
    UnknownAcquisitionCredentialError,
    UnknownMetadataCacheError,
    UnknownSpeciesApiClient,
    fetch_unknown_metadata_cache,
    format_unknown_acquisition_error,
    load_unknown_acquisition_config,
    seal_unknown_metadata_cache,
    species_query,
    verify_unknown_metadata_lock,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "unknown_acquisition.toml"


def _encoded_secret_variants(secret: str) -> tuple[str, ...]:
    quoted = urllib.parse.quote(secret, safe="")
    lowercase_percent = quoted.replace("%2B", "%2b").replace("%2F", "%2f")
    double_encoded = urllib.parse.quote(quoted, safe="")
    return (
        secret,
        quoted,
        urllib.parse.quote_plus(secret, safe=""),
        lowercase_percent,
        double_encoded,
    )


def _recording(scientific_name: str, identifier: object, **updates: object) -> dict[str, object]:
    genus, specific_epithet = scientific_name.split()
    value: dict[str, object] = {
        "id": str(identifier),
        "gen": genus,
        "sp": specific_epithet,
        "grp": "birds",
        "en": "Test bird",
        "q": "A",
    }
    value.update(updates)
    return value


def _raw_page(
    scientific_name: str,
    page: int = 1,
    *,
    num_recordings: int = 80,
    num_pages: int = 1,
    recordings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "numRecordings": str(num_recordings),
        "numSpecies": "1",
        "numPages": str(num_pages),
        "page": str(page),
        "recordings": recordings or [_recording(scientific_name, 1001)],
    }


def _canonical_page(
    scientific_name: str,
    species_index: int,
    page: int,
    *,
    num_recordings: int = 80,
    num_pages: int = 2,
    duplicate_previous_page: bool = False,
) -> dict[str, object]:
    base_page_size, remainder = divmod(num_recordings, num_pages)
    page_size = base_page_size + int(page <= remainder)
    page_offset = (page - 1) * base_page_size + min(page - 1, remainder)
    first = (species_index + 1) * 100_000 + page_offset + 1
    recordings = [
        _recording(scientific_name, identifier) for identifier in range(first, first + page_size)
    ]
    if duplicate_previous_page and page > 1:
        recordings[0]["id"] = f"XC{(species_index + 1) * 100_000 + 1}"
    return {
        "page": page,
        "num_recordings": num_recordings,
        "num_species": 1,
        "num_pages": num_pages,
        "recordings": recordings,
    }


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class UnknownAcquisitionConfigTests(unittest.TestCase):
    def test_protocol_locks_targets_queries_and_inactive_fallback(self) -> None:
        config = load_unknown_acquisition_config(CONFIG_PATH)

        self.assertEqual(
            config["candidate_pool_target_recordings_per_species"],
            CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES,
        )
        self.assertEqual(config["target_recordings_per_species"], TARGET_RECORDINGS_PER_SPECIES)
        self.assertEqual(config["api"]["endpoint"], DEFAULT_ENDPOINT)
        self.assertEqual(config["api"]["query_form"], SPECIES_QUERY_FORM)
        observed = tuple(
            (
                item["role"],
                item["active"],
                item["common_name"],
                item["scientific_name"],
                item["difficulty_group"],
            )
            for item in config["species"]
        )
        self.assertEqual(observed, LOCKED_UNKNOWN_SPECIES)
        self.assertEqual(sum(item["active"] for item in config["species"]), 5)
        self.assertEqual(config["species"][-1]["role"], "fallback")
        self.assertFalse(config["species"][-1]["active"])
        for item in config["species"]:
            genus, specific_epithet = item["scientific_name"].split()
            self.assertEqual(
                species_query(item["scientific_name"]),
                f"grp:birds gen:{genus} sp:{specific_epithet}",
            )

    def test_protocol_rejects_target_or_species_drift(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            original = CONFIG_PATH.read_text(encoding="utf-8")
            bad_target = root / "bad_target.toml"
            bad_target.write_text(
                original.replace(
                    "candidate_pool_target_recordings_per_species = 80",
                    "candidate_pool_target_recordings_per_species = 79",
                ),
                encoding="utf-8",
            )
            bad_species = root / "bad_species.toml"
            bad_species.write_text(
                original.replace("Psilopogon zeylanicus", "Psilopogon virens"),
                encoding="utf-8",
            )
            for path in (bad_target, bad_species):
                with self.subTest(path=path), self.assertRaises(UnknownAcquisitionConfigError):
                    load_unknown_acquisition_config(path)

    def test_protocol_rejects_nonfinite_numbers_and_quoted_integers(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        original = CONFIG_PATH.read_text(encoding="utf-8")
        mutations = {
            "nan_interval": (
                "request_interval_seconds = 1.0",
                "request_interval_seconds = nan",
            ),
            "infinite_timeout": (
                "timeout_seconds = 30.0",
                "timeout_seconds = inf",
            ),
            "quoted_candidate_target": (
                "candidate_pool_target_recordings_per_species = 80",
                'candidate_pool_target_recordings_per_species = "80"',
            ),
            "quoted_recording_target": (
                "target_recordings_per_species = 40",
                'target_recordings_per_species = "40"',
            ),
            "quoted_retries": (
                "maximum_retries = 5",
                'maximum_retries = "5"',
            ),
        }
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            for name, (before, after) in mutations.items():
                path = root / f"{name}.toml"
                path.write_text(original.replace(before, after), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(UnknownAcquisitionConfigError):
                    load_unknown_acquisition_config(path)


class UnknownSpeciesApiClientTests(unittest.TestCase):
    def test_public_error_formatter_redacts_raw_encoded_and_url_diagnostics(self) -> None:
        secret = "raw secret+/value"
        variants = _encoded_secret_variants(secret)
        for variant in variants:
            error = UnknownMetadataCacheError(
                f"invalid cache field {variant} at {DEFAULT_ENDPOINT}?key={variant}"
            )
            formatted = format_unknown_acquisition_error(
                error,
                environ={"XENO_CANTO_API_KEY": secret},
            )
            with self.subTest(variant=variant):
                self.assertNotIn(DEFAULT_ENDPOINT, formatted)
                for secret_variant in variants:
                    self.assertNotIn(secret_variant, formatted)

    def test_constructor_rejects_nonfinite_timing_values(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            for field in ("timeout_seconds", "request_interval_seconds"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    UnknownSpeciesApiClient("test-key", **{field: value})
        with self.assertRaises(ValueError):
            UnknownSpeciesApiClient("test-key", maximum_retries="5")  # type: ignore[arg-type]

    def test_request_uses_exact_species_query_page_and_environment_key(self) -> None:
        secret = "secret value+with symbols"
        client = UnknownSpeciesApiClient(secret)
        payload = _raw_page("Ceryle rudis")
        with patch.object(client._opener, "open", return_value=_Response(payload)) as urlopen:
            returned = client._request("Ceryle rudis", 3)

        self.assertEqual(returned, payload)
        request = urlopen.call_args.args[0]
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        self.assertEqual(parsed["query"], ["grp:birds gen:Ceryle sp:rudis"])
        self.assertEqual(parsed["page"], ["3"])
        self.assertEqual(parsed["key"], [secret])

    def test_page_validation_filters_fields_and_locks_identity(self) -> None:
        client = UnknownSpeciesApiClient("test-key")
        payload = _raw_page(
            "Ceryle rudis",
            recordings=[_recording("Ceryle rudis", 1234, unapproved="discard")],
        )
        with patch.object(client, "_request", return_value=payload):
            page = client.fetch_page("Ceryle rudis", 1)

        self.assertEqual(page["num_recordings"], 80)
        self.assertNotIn("unapproved", page["recordings"][0])
        self.assertEqual(page["recordings"][0]["id"], "1234")

        wrong_group = _raw_page(
            "Ceryle rudis",
            recordings=[_recording("Ceryle rudis", 1234, grp="mammals")],
        )
        wrong_species = _raw_page(
            "Ceryle rudis",
            recordings=[_recording("Corvus splendens", 1234)],
        )
        for invalid in (wrong_group, wrong_species):
            with (
                self.subTest(payload=invalid),
                patch.object(client, "_request", return_value=invalid),
                patch("bird_audio.unknown_acquisition.time.sleep"),
                self.assertRaises(XenoCantoApiError),
            ):
                client.fetch_page("Ceryle rudis", 1)

    def test_recording_ids_are_positive_canonical_ascii_and_collision_safe(self) -> None:
        client = UnknownSpeciesApiClient("test-key")
        canonical = _recording("Ceryle rudis", "XC123", nr="123")
        with patch.object(
            client,
            "_request",
            return_value=_raw_page("Ceryle rudis", recordings=[canonical]),
        ):
            page = client.fetch_page("Ceryle rudis", 1)
        self.assertEqual(page["recordings"][0]["id"], "123")
        self.assertEqual(page["recordings"][0]["nr"], "123")

        invalid_ids: tuple[object, ...] = (
            0,
            "0",
            "01",
            "XC0",
            "XC01",
            "xc123",
            " 123",
            "123 ",
            "١٢٣",
            "\uff11\uff12\uff13",
        )
        for invalid_id in invalid_ids:
            recording = _recording("Ceryle rudis", 123)
            recording["id"] = invalid_id
            payload = _raw_page("Ceryle rudis", recordings=[recording])
            with (
                self.subTest(invalid_id=invalid_id),
                patch.object(client, "_request", return_value=payload),
                patch("bird_audio.unknown_acquisition.time.sleep"),
                self.assertRaises(XenoCantoApiError),
            ):
                client.fetch_page("Ceryle rudis", 1)

        mismatch = _recording("Ceryle rudis", 123, nr="124")
        duplicate = [
            _recording("Ceryle rudis", "123"),
            _recording("Ceryle rudis", "XC123"),
        ]
        for recordings in ([mismatch], duplicate):
            with (
                self.subTest(recordings=recordings),
                patch.object(
                    client,
                    "_request",
                    return_value=_raw_page("Ceryle rudis", recordings=recordings),
                ),
                patch("bird_audio.unknown_acquisition.time.sleep"),
                self.assertRaises(XenoCantoApiError),
            ):
                client.fetch_page("Ceryle rudis", 1)

    def test_transient_retry_and_fatal_auth_are_distinct_and_redacted(self) -> None:
        secret = "private+key/value"
        client = UnknownSpeciesApiClient(secret, maximum_retries=1)
        retry_error = urllib.error.HTTPError(
            f"{DEFAULT_ENDPOINT}?key={urllib.parse.quote(secret)}",
            429,
            "rate limited",
            {"Retry-After": "0"},
            None,
        )
        payload = _raw_page("Corvus splendens")
        with (
            patch.object(client, "_request", side_effect=[retry_error, payload]) as request,
            patch("bird_audio.unknown_acquisition.time.sleep"),
        ):
            result = client.fetch_page("Corvus splendens", 1)
        self.assertEqual(result["page"], 1)
        self.assertEqual(request.call_count, 2)

        auth_error = urllib.error.HTTPError(
            f"{DEFAULT_ENDPOINT}?key={urllib.parse.quote(secret)}",
            401,
            "unauthorized",
            {},
            None,
        )
        with (
            patch.object(client, "_request", side_effect=auth_error),
            patch("bird_audio.unknown_acquisition.time.sleep"),
            self.assertRaises(XenoCantoFatalApiError) as caught,
        ):
            client.fetch_page("Corvus splendens", 1)
        message = str(caught.exception)
        self.assertNotIn(secret, message)
        self.assertNotIn(urllib.parse.quote(secret), message)
        self.assertNotIn("?key=", message)

    def test_raw_and_encoded_secrets_in_keys_or_values_are_rejected(self) -> None:
        secret = "raw secret+/value"
        variants = _encoded_secret_variants(secret)
        recordings: list[dict[str, object]] = []
        recordings.append(_recording("Corvus splendens", 1001, rmk=variants[0]))
        recordings.append(
            _recording(
                "Corvus splendens",
                1001,
                file=f"https://example.invalid/audio?token={variants[1]}",
            )
        )
        recordings.append(_recording("Corvus splendens", 1001, url=variants[2]))
        recordings.append(_recording("Corvus splendens", 1001, rmk=variants[3]))
        recordings.append(
            _recording(
                "Corvus splendens",
                1001,
                file=f"https://example.invalid/audio?token={variants[4]}",
            )
        )
        secret_key = _recording("Corvus splendens", 1001)
        secret_key[variants[4]] = "reflected mapping key"
        recordings.append(secret_key)

        for recording in recordings:
            client = UnknownSpeciesApiClient(secret)
            payload = _raw_page("Corvus splendens", recordings=[recording])
            with (
                self.subTest(recording=recording),
                patch.object(client, "_request", return_value=payload),
                self.assertRaises(XenoCantoApiError) as caught,
            ):
                client.fetch_page("Corvus splendens", 1)
            message = str(caught.exception)
            for variant in variants:
                self.assertNotIn(variant, message)

    def test_unexpected_request_exceptions_expose_no_key_variant(self) -> None:
        secret = "raw secret+/value"
        variants = _encoded_secret_variants(secret)
        failures = (
            ValueError(f"bad request {DEFAULT_ENDPOINT}?key={variants[1]}"),
            OSError(f"transport failed {DEFAULT_ENDPOINT}?key={variants[2]}"),
            XenoCantoApiError(f"reflected lowercase key {variants[3]}"),
            XenoCantoApiError(f"reflected double key {variants[4]}"),
        )
        for failure in failures:
            client = UnknownSpeciesApiClient(secret)
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(client, "_request", side_effect=failure),
                self.assertRaises(XenoCantoApiError) as caught,
            ):
                client.fetch_page("Corvus splendens", 1)
            message = str(caught.exception)
            self.assertNotIn(DEFAULT_ENDPOINT, message)
            for variant in variants:
                self.assertNotIn(variant, message)

    def test_redirects_and_unapproved_request_targets_are_rejected(self) -> None:
        client = UnknownSpeciesApiClient("test-key")
        handler = next(
            item
            for item in client._opener.handlers
            if isinstance(item, unknown_acquisition._RejectRedirectHandler)
        )
        with self.assertRaises(XenoCantoApiError):
            handler.redirect_request(
                urllib.request.Request(DEFAULT_ENDPOINT),
                None,
                302,
                "redirect",
                {"Location": "https://example.invalid/other"},
                "https://example.invalid/other",
            )

        client.endpoint = "https://example.invalid/api/3/recordings"
        with (
            patch.object(client._opener, "open") as open_request,
            self.assertRaises(XenoCantoApiError),
        ):
            client._request("Ceryle rudis", 1)
        open_request.assert_not_called()

    def test_recording_audio_urls_are_persisted_but_never_followed(self) -> None:
        audio_url = "https://xeno-canto.org/sounds/uploaded/example.mp3"
        payload = _raw_page(
            "Ceryle rudis",
            recordings=[_recording("Ceryle rudis", 1234, file=audio_url)],
        )
        client = UnknownSpeciesApiClient("test-key")
        with patch.object(client._opener, "open", return_value=_Response(payload)) as open_request:
            page = client.fetch_page("Ceryle rudis", 1)
        self.assertEqual(page["recordings"][0]["file"], audio_url)
        self.assertEqual(open_request.call_count, 1)
        request = open_request.call_args.args[0]
        self.assertTrue(request.full_url.startswith(f"{DEFAULT_ENDPOINT}?"))
        self.assertNotEqual(request.full_url, audio_url)


class UnknownMetadataCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_unknown_acquisition_config(CONFIG_PATH)
        self.species_index = {
            item["scientific_name"]: index for index, item in enumerate(self.config["species"])
        }
        self.boundary_counts = dict(zip(self.species_index, (39, 40, 79, 80, 81, 95), strict=True))
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=interim)
        self.root = Path(self.temporary.name)
        self.working = self.root / "unknown_working.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _page_side_effect(
        self, _client: UnknownSpeciesApiClient, scientific_name: str, page: int
    ) -> dict[str, object]:
        return _canonical_page(scientific_name, self.species_index[scientific_name], page)

    def _boundary_page_side_effect(
        self, _client: UnknownSpeciesApiClient, scientific_name: str, page: int
    ) -> dict[str, object]:
        return _canonical_page(
            scientific_name,
            self.species_index[scientific_name],
            page,
            num_recordings=self.boundary_counts[scientific_name],
        )

    def _fetch_complete(
        self,
        config_path: Path = CONFIG_PATH,
        progress_callback: unknown_acquisition.ProgressCallback | None = None,
    ) -> dict[str, object]:
        with patch.object(
            UnknownSpeciesApiClient,
            "fetch_page",
            autospec=True,
            side_effect=self._page_side_effect,
        ) as fetch_page:
            _, cache = fetch_unknown_metadata_cache(
                config_path,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
                progress_callback=progress_callback,
            )
        self.assertEqual(fetch_page.call_count, 24)
        return cache

    def test_fetches_every_page_for_primary_and_inactive_fallback_then_resumes(self) -> None:
        cache = self._fetch_complete()

        self.assertTrue(cache["complete"])
        self.assertEqual(len(cache["species"]), 6)
        self.assertFalse(cache["species"]["Streptopelia orientalis"]["active"])
        for entry in cache["species"].values():
            self.assertEqual(set(entry["pages"]), {"1", "2"})
            self.assertEqual(entry["snapshot"]["num_recordings"], 80)
            self.assertEqual(sum(page["recording_count"] for page in entry["pages"].values()), 80)
        persisted = self.working.read_text(encoding="utf-8")
        self.assertNotIn("runtime-only-secret", persisted)
        self.assertNotIn("?key=", persisted)

        with patch.object(
            UnknownSpeciesApiClient,
            "fetch_page",
            side_effect=AssertionError("completed cache must not request the API"),
        ) as fetch_page:
            _, resumed = fetch_unknown_metadata_cache(CONFIG_PATH, self.working, environ={})
        fetch_page.assert_not_called()
        self.assertTrue(resumed["complete"])

    def test_incomplete_cache_without_key_raises_credential_domain_error(self) -> None:
        with self.assertRaisesRegex(
            UnknownAcquisitionCredentialError,
            "XENO_CANTO_API_KEY",
        ):
            fetch_unknown_metadata_cache(CONFIG_PATH, self.working, environ={})

        self._fetch_complete()
        with patch.object(
            UnknownSpeciesApiClient,
            "fetch_page",
            side_effect=AssertionError("completed cache must not require a key"),
        ) as fetch_page:
            _, cache = fetch_unknown_metadata_cache(CONFIG_PATH, self.working, environ={})
        fetch_page.assert_not_called()
        self.assertTrue(cache["complete"])

    def test_mixed_inventory_boundaries_are_fetched_sealed_and_verified(self) -> None:
        config_copy = self.root / "boundary_count_config.toml"
        config_copy.write_bytes(CONFIG_PATH.read_bytes())
        events: list[dict[str, str | int]] = []
        with patch.object(
            UnknownSpeciesApiClient,
            "fetch_page",
            autospec=True,
            side_effect=self._boundary_page_side_effect,
        ) as fetch_page:
            _, cache = fetch_unknown_metadata_cache(
                config_copy,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
                progress_callback=events.append,
            )

        self.assertEqual(fetch_page.call_count, 24)
        self.assertTrue(cache["complete"])
        self.assertEqual(
            cache["candidate_pool_target_recordings_per_species"],
            CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES,
        )
        self.assertEqual(
            cache["target_recordings_per_species"],
            TARGET_RECORDINGS_PER_SPECIES,
        )
        for scientific_name, actual_count in self.boundary_counts.items():
            entry = cache["species"][scientific_name]
            self.assertEqual(entry["snapshot"]["num_recordings"], actual_count)
            self.assertEqual(
                sum(page["recording_count"] for page in entry["pages"].values()),
                actual_count,
            )
            self.assertEqual(
                entry["candidate_pool_target_recordings"],
                CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES,
            )
            self.assertEqual(entry["target_recordings"], TARGET_RECORDINGS_PER_SPECIES)
        self.assertEqual({event["phase"] for event in events[:12]}, {"fetch"})
        self.assertEqual(
            {event["phase"] for event in events[12:]},
            {"completion_revalidation"},
        )

        sealed = self.root / "boundary_count_sealed.json"
        lock = self.root / "boundary_count_lock.json"
        seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        verified = verify_unknown_metadata_lock(lock, sealed)
        self.assertTrue(verified["ready_for_candidate_planning"])
        self.assertEqual(verified["species_recording_counts"], self.boundary_counts)
        self.assertEqual(verified["recordings_total"], sum(self.boundary_counts.values()))
        self.assertEqual(
            verified["candidate_pool_target_recordings_per_species"],
            CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES,
        )
        self.assertEqual(
            verified["target_recordings_per_species"],
            TARGET_RECORDINGS_PER_SPECIES,
        )

    def test_below_pool_inventory_resumes_and_completes_revalidation(self) -> None:
        first_species = self.config["species"][0]["scientific_name"]

        def interrupted(
            _client: UnknownSpeciesApiClient, scientific_name: str, page: int
        ) -> dict[str, object]:
            if scientific_name == first_species and page == 1:
                return self._boundary_page_side_effect(_client, scientific_name, page)
            raise XenoCantoApiError("injected interruption")

        with (
            patch.object(
                UnknownSpeciesApiClient,
                "fetch_page",
                autospec=True,
                side_effect=interrupted,
            ),
            self.assertRaises(XenoCantoApiError),
        ):
            fetch_unknown_metadata_cache(
                CONFIG_PATH,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
            )
        interrupted_cache = json.loads(self.working.read_text(encoding="utf-8"))
        self.assertEqual(
            interrupted_cache["species"][first_species]["snapshot"]["num_recordings"],
            39,
        )
        self.assertEqual(
            set(interrupted_cache["species"][first_species]["pages"]),
            {"1"},
        )

        events: list[dict[str, str | int]] = []
        with patch.object(
            UnknownSpeciesApiClient,
            "fetch_page",
            autospec=True,
            side_effect=self._boundary_page_side_effect,
        ):
            _, completed = fetch_unknown_metadata_cache(
                CONFIG_PATH,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
                progress_callback=events.append,
            )
        self.assertTrue(completed["complete"])
        self.assertEqual(
            events[0],
            {
                "phase": "resume_revalidation",
                "scientific_name": first_species,
                "page": 1,
                "total_pages": 2,
            },
        )
        self.assertEqual(
            sum(event["phase"] == "completion_revalidation" for event in events),
            12,
        )

    def test_progress_callback_emits_only_safe_page_status_fields(self) -> None:
        self.working = self.root / "progress_working.json"
        events: list[dict[str, str | int]] = []
        self._fetch_complete(progress_callback=events.append)

        self.assertEqual(len(events), 24)
        self.assertEqual({event["phase"] for event in events[:12]}, {"fetch"})
        self.assertEqual(
            {event["phase"] for event in events[12:]},
            {"completion_revalidation"},
        )
        for event in events:
            self.assertEqual(
                set(event),
                {"phase", "scientific_name", "page", "total_pages"},
            )
            self.assertIn(event["scientific_name"], self.species_index)
            self.assertIn(event["page"], {1, 2})
            self.assertEqual(event["total_pages"], 2)

    def test_count_drift_stops_without_persisting_the_conflicting_page(self) -> None:
        first_species = self.config["species"][0]["scientific_name"]

        def drift(
            _client: UnknownSpeciesApiClient, scientific_name: str, page: int
        ) -> dict[str, object]:
            result = _canonical_page(
                scientific_name,
                self.species_index[scientific_name],
                page,
                num_recordings=79,
            )
            if scientific_name == first_species and page == 2:
                result["num_recordings"] = 78
            return result

        with (
            patch.object(
                UnknownSpeciesApiClient,
                "fetch_page",
                autospec=True,
                side_effect=drift,
            ),
            self.assertRaises(XenoCantoApiError) as caught,
        ):
            fetch_unknown_metadata_cache(
                CONFIG_PATH,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
            )
        persisted = json.loads(self.working.read_text(encoding="utf-8"))
        self.assertFalse(persisted["complete"])
        self.assertEqual(set(persisted["species"][first_species]["pages"]), {"1"})
        self.assertIn("fresh --working-cache path", str(caught.exception))
        self.assertIn("Retain this working cache as evidence", str(caught.exception))

    def test_cross_page_duplicate_stops_before_checkpoint(self) -> None:
        first_species = self.config["species"][0]["scientific_name"]

        def duplicate(
            _client: UnknownSpeciesApiClient, scientific_name: str, page: int
        ) -> dict[str, object]:
            return _canonical_page(
                scientific_name,
                self.species_index[scientific_name],
                page,
                num_recordings=79,
                duplicate_previous_page=scientific_name == first_species and page == 2,
            )

        with (
            patch.object(
                UnknownSpeciesApiClient,
                "fetch_page",
                autospec=True,
                side_effect=duplicate,
            ),
            self.assertRaises(XenoCantoApiError),
        ):
            fetch_unknown_metadata_cache(
                CONFIG_PATH,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
            )
        persisted = json.loads(self.working.read_text(encoding="utf-8"))
        self.assertEqual(set(persisted["species"][first_species]["pages"]), {"1"})

    def test_secret_variants_abort_before_any_page_checkpoint(self) -> None:
        secret = "raw secret+/value"
        variants = _encoded_secret_variants(secret)
        cases: list[tuple[str, str | None, str]] = [
            ("raw_text", "rmk", variants[0]),
            ("quoted_url", "file", f"https://example.invalid/?token={variants[1]}"),
            ("quote_plus_url", "url", f"https://example.invalid/?token={variants[2]}"),
            ("lowercase_percent", "rmk", variants[3]),
            ("double_encoded", "file", f"https://example.invalid/?token={variants[4]}"),
            ("mapping_key", None, variants[4]),
        ]
        first_species = self.config["species"][0]["scientific_name"]
        for name, field, value in cases:
            working = self.root / f"secret_{name}.json"

            def reflected(
                _client: UnknownSpeciesApiClient,
                scientific_name: str,
                page: int,
                field_to_set: str | None = field,
                value_to_set: str = value,
            ) -> dict[str, object]:
                result = _canonical_page(scientific_name, self.species_index[scientific_name], page)
                if scientific_name == first_species and page == 1:
                    recording = result["recordings"][0]  # type: ignore[index]
                    if field_to_set is None:
                        recording[value_to_set] = "reflected key"
                    else:
                        recording[field_to_set] = value_to_set
                return result

            with (
                self.subTest(name=name),
                patch.object(
                    UnknownSpeciesApiClient,
                    "fetch_page",
                    autospec=True,
                    side_effect=reflected,
                ),
                self.assertRaises(XenoCantoApiError) as caught,
            ):
                fetch_unknown_metadata_cache(
                    CONFIG_PATH,
                    working,
                    environ={"XENO_CANTO_API_KEY": secret},
                )
            persisted = working.read_text(encoding="utf-8")
            page_cache = json.loads(persisted)["species"][first_species]["pages"]
            self.assertEqual(page_cache, {})
            for variant in variants:
                self.assertNotIn(variant, persisted)
                self.assertNotIn(variant, str(caught.exception))

    def test_resume_refuses_same_count_page_substitution_before_append(self) -> None:
        first_species = self.config["species"][0]["scientific_name"]
        for mutation in ("identifier", "metadata"):
            working = self.root / f"interrupted_{mutation}.json"

            def interrupted(
                _client: UnknownSpeciesApiClient,
                scientific_name: str,
                page: int,
            ) -> dict[str, object]:
                if scientific_name == first_species and page == 1:
                    return _canonical_page(
                        scientific_name, self.species_index[scientific_name], page
                    )
                raise XenoCantoApiError("injected interruption")

            with (
                patch.object(
                    UnknownSpeciesApiClient,
                    "fetch_page",
                    autospec=True,
                    side_effect=interrupted,
                ),
                self.assertRaises(XenoCantoApiError),
            ):
                fetch_unknown_metadata_cache(
                    CONFIG_PATH,
                    working,
                    environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
                )
            before_resume = working.read_bytes()

            def substituted(
                _client: UnknownSpeciesApiClient,
                scientific_name: str,
                page: int,
                selected_mutation: str = mutation,
            ) -> dict[str, object]:
                result = _canonical_page(scientific_name, self.species_index[scientific_name], page)
                if selected_mutation == "identifier":
                    result["recordings"][0]["id"] = "99999999"  # type: ignore[index]
                else:
                    result["recordings"][0]["q"] = "B"  # type: ignore[index]
                return result

            events: list[dict[str, str | int]] = []
            with (
                self.subTest(mutation=mutation),
                patch.object(
                    UnknownSpeciesApiClient,
                    "fetch_page",
                    autospec=True,
                    side_effect=substituted,
                ) as fetch_page,
                self.assertRaises(XenoCantoApiError) as caught,
            ):
                fetch_unknown_metadata_cache(
                    CONFIG_PATH,
                    working,
                    environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
                    progress_callback=events.append,
                )
            self.assertEqual(fetch_page.call_count, 1)
            self.assertEqual(working.read_bytes(), before_resume)
            self.assertEqual(
                events,
                [
                    {
                        "phase": "resume_revalidation",
                        "scientific_name": first_species,
                        "page": 1,
                        "total_pages": 2,
                    }
                ],
            )
            self.assertIn("fresh --working-cache path", str(caught.exception))

    def test_noncanonical_page_mapping_key_is_rejected_cleanly(self) -> None:
        first_species = self.config["species"][0]["scientific_name"]

        def interrupted(
            _client: UnknownSpeciesApiClient, scientific_name: str, page: int
        ) -> dict[str, object]:
            if scientific_name == first_species and page == 1:
                return _canonical_page(scientific_name, self.species_index[scientific_name], page)
            raise XenoCantoApiError("injected interruption")

        with (
            patch.object(
                UnknownSpeciesApiClient,
                "fetch_page",
                autospec=True,
                side_effect=interrupted,
            ),
            self.assertRaises(XenoCantoApiError),
        ):
            fetch_unknown_metadata_cache(
                CONFIG_PATH,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
            )
        cache = json.loads(self.working.read_text(encoding="utf-8"))
        pages = cache["species"][first_species]["pages"]
        pages["01"] = pages.pop("1")
        self.working.write_text(json.dumps(cache), encoding="utf-8")
        with (
            patch.object(UnknownSpeciesApiClient, "fetch_page") as fetch_page,
            self.assertRaises(UnknownMetadataCacheError),
        ):
            fetch_unknown_metadata_cache(
                CONFIG_PATH,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
            )
        fetch_page.assert_not_called()

    def test_duplicate_recording_id_across_species_is_rejected_before_checkpoint(self) -> None:
        first_species = self.config["species"][0]["scientific_name"]
        second_species = self.config["species"][1]["scientific_name"]

        def overlapping(
            _client: UnknownSpeciesApiClient, scientific_name: str, page: int
        ) -> dict[str, object]:
            result = _canonical_page(
                scientific_name,
                self.species_index[scientific_name],
                page,
                num_pages=1,
            )
            if scientific_name == second_species:
                first_id = (self.species_index[first_species] + 1) * 100_000 + 1
                result["recordings"][0]["id"] = f"XC{first_id}"  # type: ignore[index]
            return result

        with (
            patch.object(
                UnknownSpeciesApiClient,
                "fetch_page",
                autospec=True,
                side_effect=overlapping,
            ),
            self.assertRaises(XenoCantoApiError),
        ):
            fetch_unknown_metadata_cache(
                CONFIG_PATH,
                self.working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
            )
        cache = json.loads(self.working.read_text(encoding="utf-8"))
        self.assertEqual(set(cache["species"][first_species]["pages"]), {"1"})
        self.assertEqual(cache["species"][second_species]["pages"], {})

    def test_known_species_universe_is_bound_and_overlap_is_rejected(self) -> None:
        parsed = unknown_acquisition.tomllib.loads(
            (PROJECT_ROOT / "configs" / "data.toml").read_text(encoding="utf-8")
        )
        parsed["known_species"][0]["scientific_name"] = self.config["species"][0]["scientific_name"]
        with (
            patch.object(unknown_acquisition.tomllib, "loads", return_value=parsed),
            self.assertRaises(UnknownAcquisitionConfigError),
        ):
            unknown_acquisition._load_known_species_snapshot(self.config)

    def test_data_config_unknown_drift_blocks_discovery_and_sealing(self) -> None:
        data_text = (PROJECT_ROOT / "configs" / "data.toml").read_text(encoding="utf-8")
        original_loads = unknown_acquisition.tomllib.loads

        def parser_for(mutated_data: dict[str, object]):
            def parse(text: str) -> dict[str, object]:
                if 'raw_audio_dir = "dataset"' in text:
                    return copy.deepcopy(mutated_data)
                return original_loads(text)

            return parse

        target_drift = original_loads(data_text)
        target_drift["unknown_species"][0]["target_recordings"] = 41
        discovery_working = self.root / "drift_discovery.json"
        with (
            patch.object(
                unknown_acquisition.tomllib,
                "loads",
                side_effect=parser_for(target_drift),
            ),
            self.assertRaises(UnknownAcquisitionConfigError),
        ):
            fetch_unknown_metadata_cache(
                CONFIG_PATH,
                discovery_working,
                environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
            )
        self.assertFalse(discovery_working.exists())

        self._fetch_complete()
        group_drift = original_loads(data_text)
        group_drift["unknown_species"][0]["difficulty_group"] = "other_family"
        sealed = self.root / "drift_sealed.json"
        lock = self.root / "drift_lock.json"
        with (
            patch.object(
                unknown_acquisition.tomllib,
                "loads",
                side_effect=parser_for(group_drift),
            ),
            self.assertRaises(UnknownAcquisitionConfigError),
        ):
            seal_unknown_metadata_cache(CONFIG_PATH, self.working, sealed, lock)
        self.assertFalse(sealed.exists())
        self.assertFalse(lock.exists())

    def test_seal_and_verify_bind_config_working_snapshot_and_sealed_cache(self) -> None:
        config_copy = self.root / "unknown_acquisition.toml"
        config_copy.write_bytes(CONFIG_PATH.read_bytes())
        self._fetch_complete(config_copy)
        sealed = self.root / "unknown_sealed.json"
        lock = self.root / "unknown_lock.json"

        _, _, lock_payload = seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        verified = verify_unknown_metadata_lock(lock, sealed)
        self.assertTrue(verified["ready_for_candidate_planning"])
        self.assertEqual(verified["species_count"], 6)
        self.assertEqual(verified["primary_species_count"], 5)
        self.assertEqual(verified["inactive_fallback_count"], 1)
        self.assertEqual(verified["known_species_count"], 15)
        self.assertEqual(verified["recordings_total"], 480)
        self.assertEqual(
            verified["artifacts"]["known_species_config"]["path"],
            "configs/data.toml",
        )
        self.assertEqual(
            verified["source_working_cache_sha256"],
            lock_payload["source_working_cache_sha256"],
        )

        original_lock = lock.read_bytes()
        lock_mutations = []
        extra_field = json.loads(original_lock.decode("utf-8"))
        extra_field["unexpected"] = "not permitted"
        lock_mutations.append(extra_field)
        extra_artifact = json.loads(original_lock.decode("utf-8"))
        extra_artifact["artifacts"]["unexpected"] = {
            "path": "configs/data.toml",
            "sha256": extra_artifact["known_species_config_sha256"],
        }
        lock_mutations.append(extra_artifact)
        inconsistent_working_hash = json.loads(original_lock.decode("utf-8"))
        inconsistent_working_hash["artifacts"]["working_cache"]["sha256"] = "0" * 64
        lock_mutations.append(inconsistent_working_hash)
        for mutation in lock_mutations:
            lock.write_text(json.dumps(mutation), encoding="utf-8")
            with self.assertRaises(UnknownMetadataCacheError):
                verify_unknown_metadata_lock(lock, sealed)
            lock.write_bytes(original_lock)

        sealed_payload = json.loads(sealed.read_text(encoding="utf-8"))
        sealed_payload["complete"] = False
        sealed.write_text(json.dumps(sealed_payload), encoding="utf-8")
        with self.assertRaises(UnknownMetadataCacheError):
            verify_unknown_metadata_lock(lock, sealed)

    def test_sealed_publication_is_create_only_under_a_destination_race(self) -> None:
        config_copy = self.root / "race_config.toml"
        config_copy.write_bytes(CONFIG_PATH.read_bytes())
        self._fetch_complete(config_copy)
        sealed = self.root / "race_sealed.json"
        lock = self.root / "race_lock.json"
        competing_bytes = b"competitor-owned-bytes\n"
        original_create = unknown_acquisition._create_json_exclusive

        def competing_create(path: str | Path, value: object) -> Path:
            destination = Path(path)
            if destination == sealed:
                destination.write_bytes(competing_bytes)
            return original_create(destination, value)

        with (
            patch.object(
                unknown_acquisition,
                "_create_json_exclusive",
                side_effect=competing_create,
            ),
            self.assertRaises(FileExistsError),
        ):
            seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        self.assertEqual(sealed.read_bytes(), competing_bytes)
        self.assertFalse(lock.exists())

    def test_sealed_bytes_are_rechecked_immediately_before_lock_publication(self) -> None:
        config_copy = self.root / "boundary_config.toml"
        config_copy.write_bytes(CONFIG_PATH.read_bytes())
        self._fetch_complete(config_copy)
        sealed = self.root / "boundary_sealed.json"
        lock = self.root / "boundary_lock.json"
        original_summary = unknown_acquisition._cache_summary

        def mutate_after_hash(cache: dict[str, object]) -> dict[str, object]:
            summary = original_summary(cache)
            sealed.write_bytes(b"adversarial replacement\n")
            return summary

        with (
            patch.object(
                unknown_acquisition,
                "_cache_summary",
                side_effect=mutate_after_hash,
            ),
            self.assertRaises(RuntimeError),
        ):
            seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        self.assertEqual(sealed.read_bytes(), b"adversarial replacement\n")
        self.assertFalse(lock.exists())

    def test_lock_failure_recovers_idempotently_without_rewriting_sealed_bytes(self) -> None:
        config_copy = self.root / "recovery_config.toml"
        config_copy.write_bytes(CONFIG_PATH.read_bytes())
        self._fetch_complete(config_copy)
        sealed = self.root / "recovery_sealed.json"
        lock = self.root / "recovery_lock.json"
        original_create = unknown_acquisition._create_json_exclusive

        def fail_lock(path: str | Path, value: object) -> Path:
            destination = Path(path)
            if destination == lock:
                raise RuntimeError("injected lock publication failure")
            return original_create(destination, value)

        with (
            patch.object(
                unknown_acquisition,
                "_create_json_exclusive",
                side_effect=fail_lock,
            ),
            self.assertRaises(RuntimeError),
        ):
            seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        sealed_bytes = sealed.read_bytes()
        self.assertFalse(lock.exists())

        self.working.write_text("{}\n", encoding="utf-8")
        _, _, recovered = seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        self.assertEqual(sealed.read_bytes(), sealed_bytes)
        self.assertTrue(verify_unknown_metadata_lock(lock, sealed)["ready_for_candidate_planning"])
        self.assertEqual(
            recovered["artifacts"]["working_cache"]["sha256"],
            recovered["source_working_cache_sha256"],
        )

        _, _, idempotent = seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        self.assertEqual(idempotent, recovered)
        self.assertEqual(sealed.read_bytes(), sealed_bytes)

        with tempfile.TemporaryDirectory() as outside, self.assertRaises(ValueError):
            verify_unknown_metadata_lock(lock, Path(outside) / "expected.json")

    def test_orphan_recovery_rejects_a_forged_source_working_hash(self) -> None:
        config_copy = self.root / "forged_config.toml"
        config_copy.write_bytes(CONFIG_PATH.read_bytes())
        self._fetch_complete(config_copy)
        sealed = self.root / "forged_sealed.json"
        lock = self.root / "forged_lock.json"
        original_create = unknown_acquisition._create_json_exclusive

        def fail_lock(path: str | Path, value: object) -> Path:
            destination = Path(path)
            if destination == lock:
                raise RuntimeError("injected lock publication failure")
            return original_create(destination, value)

        with (
            patch.object(
                unknown_acquisition,
                "_create_json_exclusive",
                side_effect=fail_lock,
            ),
            self.assertRaises(RuntimeError),
        ):
            seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)

        payload = json.loads(sealed.read_text(encoding="utf-8"))
        payload["source_working_cache_sha256"] = "0" * 64
        sealed.write_bytes(unknown_acquisition._canonical_json_bytes(payload))
        self.working.unlink()
        with self.assertRaises(UnknownMetadataCacheError) as caught:
            seal_unknown_metadata_cache(config_copy, self.working, sealed, lock)
        self.assertIn("not reproducible", str(caught.exception))
        self.assertFalse(lock.exists())

    def test_outside_project_inputs_are_rejected_before_output_side_effects(self) -> None:
        sealed = self.root / "outside_sealed.json"
        lock = self.root / "outside_lock.json"
        fetch_output = self.root / "outside_fetch.json"
        with tempfile.TemporaryDirectory() as outside:
            outside_root = Path(outside)
            outside_config = outside_root / "config.toml"
            outside_config.write_bytes(CONFIG_PATH.read_bytes())
            outside_working = outside_root / "working.json"
            outside_working.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                fetch_unknown_metadata_cache(
                    outside_config,
                    fetch_output,
                    environ={"XENO_CANTO_API_KEY": "runtime-only-secret"},
                )
            self.assertFalse(fetch_output.exists())
            with self.assertRaises(ValueError):
                seal_unknown_metadata_cache(CONFIG_PATH, outside_working, sealed, lock)
            self.assertFalse(sealed.exists())
            self.assertFalse(lock.exists())
            with self.assertRaises(ValueError):
                verify_unknown_metadata_lock(outside_root / "lock.json")


if __name__ == "__main__":
    unittest.main()
