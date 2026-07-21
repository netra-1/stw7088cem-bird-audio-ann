from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bird_audio.hashing import sha256_bytes, sha256_file, sha256_json
from bird_audio.io_utils import atomic_write_json, require_unchanged
from bird_audio.locking import project_lock
from bird_audio.metadata import (
    API_VERSION,
    DEFAULT_ENDPOINT,
    PERSISTED_RECORDING_FIELDS,
    XenoCantoApiError,
    XenoCantoFatalApiError,
    _clean_error,
    _retry_after_seconds,
)
from bird_audio.paths import (
    PROJECT_ROOT,
    is_relative_to,
    require_safe_output,
    resolve_project_path,
)

UNKNOWN_ACQUISITION_CONFIG_SCHEMA_VERSION = "1.0"
UNKNOWN_METADATA_CACHE_SCHEMA_VERSION = "1.0"
UNKNOWN_METADATA_LOCK_SCHEMA_VERSION = "1.0"
SPECIES_QUERY_FORM = "grp:birds gen:<genus> sp:<specific_epithet>"
CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES = 80
TARGET_RECORDINGS_PER_SPECIES = 40
KNOWN_SPECIES_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.toml"
MAXIMUM_SECRET_DECODE_DEPTH = 2

LOCKED_UNKNOWN_SPECIES = (
    (
        "primary",
        True,
        "Brown-headed Barbet",
        "Psilopogon zeylanicus",
        "family_matched",
    ),
    ("primary", True, "Jungle Myna", "Acridotheres fuscus", "family_matched"),
    ("primary", True, "Pied Kingfisher", "Ceryle rudis", "family_matched"),
    ("primary", True, "House Crow", "Corvus splendens", "other_family"),
    ("primary", True, "Grey Francolin", "Ortygornis pondicerianus", "other_family"),
    (
        "fallback",
        False,
        "Oriental Turtle Dove",
        "Streptopelia orientalis",
        "fallback",
    ),
)

_CONFIG_KEYS = {
    "schema_version",
    "candidate_pool_target_recordings_per_species",
    "target_recordings_per_species",
    "api",
    "species",
}
_API_CONFIG_KEYS = {
    "version",
    "endpoint",
    "query_form",
    "api_key_environment",
    "request_interval_seconds",
    "timeout_seconds",
    "maximum_retries",
    "pagination_policy",
    "snapshot_consistency",
    "duplicate_recording_policy",
}
_SPECIES_CONFIG_KEYS = {
    "role",
    "active",
    "common_name",
    "scientific_name",
    "difficulty_group",
}
_WORKING_CACHE_KEYS = {
    "schema_version",
    "api_version",
    "endpoint",
    "query_form",
    "config_path",
    "config_sha256",
    "known_species_config_path",
    "known_species_config_sha256",
    "known_species_set_sha256",
    "known_species_count",
    "candidate_pool_target_recordings_per_species",
    "target_recordings_per_species",
    "created_at_utc",
    "updated_at_utc",
    "completed_at_utc",
    "complete",
    "species",
}
_SEALED_CACHE_KEYS = {
    *_WORKING_CACHE_KEYS,
    "sealed",
    "sealed_at_utc",
    "source_working_cache_path",
    "source_working_cache_sha256",
}
_CACHE_SPECIES_KEYS = {
    "role",
    "active",
    "common_name",
    "scientific_name",
    "difficulty_group",
    "candidate_pool_target_recordings",
    "target_recordings",
    "query",
    "snapshot",
    "pages",
}
_SNAPSHOT_KEYS = {"num_recordings", "num_species", "num_pages"}
_PAGE_KEYS = {
    "fetched_at_utc",
    "page",
    "num_recordings",
    "num_species",
    "num_pages",
    "recording_count",
    "recording_ids_sha256",
    "recordings",
}
_CANONICAL_PAGE_KEYS = {
    "page",
    "num_recordings",
    "num_species",
    "num_pages",
    "recordings",
}
_LOCK_KEYS = {
    "schema_version",
    "locked_at_utc",
    "ready_for_candidate_planning",
    "api_version",
    "endpoint",
    "query_form",
    "pagination_policy",
    "snapshot_consistency",
    "species_count",
    "primary_species_count",
    "fallback_species_count",
    "inactive_fallback_count",
    "candidate_pool_target_recordings_per_species",
    "target_recordings_per_species",
    "config_sha256",
    "known_species_config_sha256",
    "known_species_set_sha256",
    "known_species_count",
    "source_working_cache_sha256",
    "sealed_cache_sha256",
    "species_recording_counts",
    "recordings_total",
    "recording_set_sha256",
    "species_snapshot_sha256",
    "artifacts",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STRICT_BINOMIAL_PATTERN = re.compile(r"^[A-Z][A-Za-z]+ [a-z][A-Za-z]+$")
_URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_SEAL_ONLY_FIELDS = {
    "sealed",
    "sealed_at_utc",
    "source_working_cache_path",
    "source_working_cache_sha256",
}

ProgressCallback = Callable[[dict[str, str | int]], None]


class UnknownAcquisitionConfigError(ValueError):
    pass


class UnknownMetadataCacheError(ValueError):
    pass


class UnknownAcquisitionCredentialError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    if not is_relative_to(resolved, PROJECT_ROOT):
        raise ValueError(f"Protocol artifact must be inside the project: {resolved}")
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def _require_project_path(path: str | Path, context: str) -> Path:
    resolved = resolve_project_path(path)
    if not is_relative_to(resolved, PROJECT_ROOT):
        raise ValueError(f"{context} must be inside the project")
    return resolved


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], context: str, error_type: type[ValueError]
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise error_type(f"{context} fields are invalid ({', '.join(details)})")


def _strict_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        number = int(value)
    else:
        raise ValueError(f"{context} must be a non-negative integer")
    if number < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return number


def _strict_positive_int(value: Any, context: str) -> int:
    number = _strict_nonnegative_int(value, context)
    if number < 1:
        raise ValueError(f"{context} must be a positive integer")
    return number


def _strict_config_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative TOML integer")
    return value


def _strict_config_positive_int(value: Any, context: str) -> int:
    number = _strict_config_nonnegative_int(value, context)
    if number < 1:
        raise ValueError(f"{context} must be a positive TOML integer")
    return number


def _strict_positive_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{context} must be a positive number")
    return number


def _validate_config(config: dict[str, Any]) -> None:
    try:
        _require_exact_keys(config, _CONFIG_KEYS, "unknown acquisition config", ValueError)
        if config["schema_version"] != UNKNOWN_ACQUISITION_CONFIG_SCHEMA_VERSION:
            raise ValueError("unknown acquisition config schema is not supported")
        if (
            _strict_config_positive_int(
                config["candidate_pool_target_recordings_per_species"],
                "candidate pool target",
            )
            != CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES
        ):
            raise ValueError("candidate pool target must be exactly 80 recordings per species")
        if (
            _strict_config_positive_int(config["target_recordings_per_species"], "recording target")
            != TARGET_RECORDINGS_PER_SPECIES
        ):
            raise ValueError("recording target must be exactly 40 recordings per species")

        api = config["api"]
        if not isinstance(api, dict):
            raise ValueError("api must be a table")
        _require_exact_keys(api, _API_CONFIG_KEYS, "unknown acquisition api", ValueError)
        if api["version"] != API_VERSION:
            raise ValueError("API version must be xeno-canto API v3")
        if api["endpoint"] != DEFAULT_ENDPOINT:
            raise ValueError("only the approved HTTPS Xeno-canto API v3 endpoint is permitted")
        if api["query_form"] != SPECIES_QUERY_FORM:
            raise ValueError("species query form is not the locked exact-birds form")
        if api["api_key_environment"] != "XENO_CANTO_API_KEY":
            raise ValueError("the API key must be sourced from XENO_CANTO_API_KEY")
        if _strict_positive_float(api["request_interval_seconds"], "request interval") < 1.0:
            raise ValueError("request interval must be at least one second")
        _strict_positive_float(api["timeout_seconds"], "request timeout")
        maximum_retries = _strict_config_nonnegative_int(api["maximum_retries"], "maximum retries")
        if maximum_retries > 10:
            raise ValueError("maximum retries cannot exceed 10")
        if api["pagination_policy"] != "all_server_reported_pages_in_ascending_order":
            raise ValueError("pagination policy is not the locked sequential policy")
        if api["snapshot_consistency"] != "exact_numRecordings_numSpecies_numPages_and_page":
            raise ValueError("snapshot consistency policy is not locked")
        if api["duplicate_recording_policy"] != "fatal":
            raise ValueError("duplicate recording policy must be fatal")

        species = config["species"]
        if not isinstance(species, list):
            raise ValueError("species must be an array of tables")
        observed: list[tuple[Any, ...]] = []
        for index, entry in enumerate(species):
            if not isinstance(entry, dict):
                raise ValueError(f"species[{index}] must be a table")
            _require_exact_keys(entry, _SPECIES_CONFIG_KEYS, f"species[{index}]", ValueError)
            if not isinstance(entry["active"], bool):
                raise ValueError(f"species[{index}].active must be Boolean")
            scientific_name = str(entry["scientific_name"])
            if not _STRICT_BINOMIAL_PATTERN.fullmatch(scientific_name):
                raise ValueError(f"species[{index}].scientific_name is not a strict binomial")
            observed.append(
                (
                    entry["role"],
                    entry["active"],
                    entry["common_name"],
                    scientific_name,
                    entry["difficulty_group"],
                )
            )
        if tuple(observed) != LOCKED_UNKNOWN_SPECIES:
            raise ValueError("species identities, ordering, roles, or active states are not locked")
    except ValueError as exc:
        if isinstance(exc, UnknownAcquisitionConfigError):
            raise
        raise UnknownAcquisitionConfigError(str(exc)) from None


def _load_config_snapshot(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    config_path = resolve_project_path(path)
    if not is_relative_to(config_path, PROJECT_ROOT):
        raise UnknownAcquisitionConfigError("unknown acquisition config must be inside the project")
    payload = config_path.read_bytes()
    digest = sha256_bytes(payload)
    try:
        config = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise UnknownAcquisitionConfigError(
            "unknown acquisition config is not valid UTF-8 TOML"
        ) from exc
    if not isinstance(config, dict):
        raise UnknownAcquisitionConfigError("unknown acquisition config is not a TOML table")
    _validate_config(config)
    return config_path, config, digest


def _load_known_species_snapshot(
    unknown_config: dict[str, Any],
    path: str | Path = KNOWN_SPECIES_CONFIG_PATH,
) -> dict[str, Any]:
    try:
        config_path = _require_project_path(path, "known-species config")
        if config_path != KNOWN_SPECIES_CONFIG_PATH.resolve():
            raise ValueError("known-species config must be configs/data.toml")
        payload = config_path.read_bytes()
        config = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise UnknownAcquisitionConfigError("known-species config is not valid UTF-8 TOML") from exc
    except ValueError as exc:
        raise UnknownAcquisitionConfigError(str(exc)) from None
    known_species = config.get("known_species") if isinstance(config, dict) else None
    if not isinstance(known_species, list) or len(known_species) != 15:
        raise UnknownAcquisitionConfigError("known-species config must define exactly 15 species")
    data_unknown = config.get("unknown_species")
    data_fallback = config.get("fallback_unknown_species")
    if not isinstance(data_unknown, list) or not isinstance(data_fallback, list):
        raise UnknownAcquisitionConfigError("data config unknown-species declarations are invalid")
    observed_unknown: list[tuple[Any, ...]] = []
    for index, entry in enumerate(data_unknown):
        if not isinstance(entry, dict):
            raise UnknownAcquisitionConfigError(f"data unknown_species[{index}] is invalid")
        target = entry.get("target_recordings")
        if isinstance(target, bool) or not isinstance(target, int) or target != 40:
            raise UnknownAcquisitionConfigError(
                f"data unknown_species[{index}] target must be the integer 40"
            )
        observed_unknown.append(
            (
                "primary",
                True,
                entry.get("common_name"),
                entry.get("scientific_name"),
                entry.get("difficulty_group"),
            )
        )
    for index, entry in enumerate(data_fallback):
        if not isinstance(entry, dict):
            raise UnknownAcquisitionConfigError(
                f"data fallback_unknown_species[{index}] is invalid"
            )
        target = entry.get("target_recordings")
        if isinstance(target, bool) or not isinstance(target, int) or target != 40:
            raise UnknownAcquisitionConfigError(
                f"data fallback_unknown_species[{index}] target must be the integer 40"
            )
        observed_unknown.append(
            (
                "fallback",
                False,
                entry.get("common_name"),
                entry.get("scientific_name"),
                "fallback",
            )
        )
    if tuple(observed_unknown) != LOCKED_UNKNOWN_SPECIES:
        raise UnknownAcquisitionConfigError(
            "data config unknown species, order, roles, or difficulty groups are not locked"
        )
    canonical: list[dict[str, str]] = []
    for index, entry in enumerate(known_species):
        if not isinstance(entry, dict):
            raise UnknownAcquisitionConfigError(f"known_species[{index}] is invalid")
        common_name = entry.get("common_name")
        scientific_name = entry.get("scientific_name")
        if not isinstance(common_name, str) or not common_name.strip():
            raise UnknownAcquisitionConfigError(f"known_species[{index}].common_name is invalid")
        if not isinstance(scientific_name, str) or not _STRICT_BINOMIAL_PATTERN.fullmatch(
            scientific_name
        ):
            raise UnknownAcquisitionConfigError(
                f"known_species[{index}].scientific_name is invalid"
            )
        canonical.append({"common_name": common_name, "scientific_name": scientific_name})
    common_names = [entry["common_name"].casefold() for entry in canonical]
    scientific_names = [entry["scientific_name"].casefold() for entry in canonical]
    if len(set(common_names)) != len(canonical) or len(set(scientific_names)) != len(canonical):
        raise UnknownAcquisitionConfigError("known-species identities must be unique")
    unknown_scientific_names = {
        str(entry["scientific_name"]).casefold() for entry in unknown_config["species"]
    }
    unknown_common_names = {
        str(entry["common_name"]).casefold() for entry in unknown_config["species"]
    }
    if unknown_scientific_names.intersection(scientific_names) or unknown_common_names.intersection(
        common_names
    ):
        raise UnknownAcquisitionConfigError("known and unknown species must not overlap")
    canonical.sort(key=lambda entry: entry["scientific_name"])
    return {
        "path": config_path,
        "sha256": sha256_bytes(payload),
        "set_sha256": sha256_json(canonical),
        "count": len(canonical),
        "scientific_names": tuple(entry["scientific_name"] for entry in canonical),
    }


def load_unknown_acquisition_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the unknown-species acquisition protocol."""
    _, config, _ = _load_config_snapshot(path)
    return copy.deepcopy(config)


def species_query(scientific_name: str) -> str:
    if not _STRICT_BINOMIAL_PATTERN.fullmatch(scientific_name):
        raise ValueError("scientific_name must be a strict two-part binomial")
    genus, specific_epithet = scientific_name.split()
    return f"grp:birds gen:{genus} sp:{specific_epithet}"


def _secret_variants(secret: str) -> tuple[str, ...]:
    return tuple(
        value
        for value in dict.fromkeys(
            (
                secret,
                urllib.parse.quote(secret, safe=""),
                urllib.parse.quote_plus(secret, safe=""),
            )
        )
        if value
    )


def _decoded_text_forms(value: str) -> set[str]:
    forms = {value}
    frontier = {value}
    for _ in range(MAXIMUM_SECRET_DECODE_DEPTH):
        next_frontier: set[str] = set()
        for item in frontier:
            next_frontier.add(urllib.parse.unquote(item))
            next_frontier.add(urllib.parse.unquote_plus(item))
        next_frontier.difference_update(forms)
        if not next_frontier:
            break
        forms.update(next_frontier)
        frontier = next_frontier
    return forms


def _contains_request_secret(value: Any, secret: str) -> bool:
    variants = _secret_variants(secret)

    def contains(item: Any) -> bool:
        if isinstance(item, Mapping):
            return any(contains(key) or contains(nested) for key, nested in item.items())
        if isinstance(item, (list, tuple, set, frozenset)):
            return any(contains(nested) for nested in item)
        return any(
            variant in form for form in _decoded_text_forms(str(item)) for variant in variants
        )

    return bool(variants and contains(value))


def _strict_recording_id(recording: dict[str, Any], context: str) -> str:
    values: list[str] = []
    for field in ("id", "nr"):
        raw = recording.get(field)
        if raw in (None, ""):
            continue
        if isinstance(raw, bool):
            raise XenoCantoApiError(f"{context} has an invalid recording identifier")
        if isinstance(raw, int):
            value = str(raw)
        elif isinstance(raw, str):
            value = raw
        else:
            raise XenoCantoApiError(f"{context} has an invalid recording identifier")
        if value.startswith("XC"):
            value = value[2:]
        if not re.fullmatch(r"[1-9][0-9]*", value):
            raise XenoCantoApiError(f"{context} has an invalid recording identifier")
        values.append(value)
    if not values:
        raise XenoCantoApiError(f"{context} is missing a recording identifier")
    if len(set(values)) != 1:
        raise XenoCantoApiError(f"{context} has conflicting recording identifiers")
    return values[0]


def _validate_recording_identity(
    recording: dict[str, Any], scientific_name: str, context: str
) -> str:
    genus, specific_epithet = scientific_name.split()
    if str(recording.get("gen") or "").strip() != genus:
        raise XenoCantoApiError(f"{context} genus does not match {scientific_name}")
    if str(recording.get("sp") or "").strip() != specific_epithet:
        raise XenoCantoApiError(f"{context} species does not match {scientific_name}")
    group_values = [
        str(recording[field]).strip().casefold()
        for field in ("grp", "group")
        if recording.get(field) not in (None, "")
    ]
    if not group_values or any(value != "birds" for value in group_values):
        raise XenoCantoApiError(f"{context} is not explicitly in the birds group")
    return _strict_recording_id(recording, context)


def _canonical_recordings(
    recordings: Any, scientific_name: str, context: str
) -> list[dict[str, Any]]:
    if not isinstance(recordings, list) or not recordings:
        raise XenoCantoApiError(f"{context} recordings must be a non-empty list")
    persisted: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for index, recording in enumerate(recordings):
        recording_context = f"{context} recording[{index}]"
        if not isinstance(recording, dict):
            raise XenoCantoApiError(f"{recording_context} is not an object")
        identifier = _validate_recording_identity(recording, scientific_name, recording_context)
        identifiers.append(identifier)
        persisted_recording = {
            key: value for key, value in recording.items() if key in PERSISTED_RECORDING_FIELDS
        }
        for identifier_field in ("id", "nr"):
            if identifier_field in persisted_recording:
                persisted_recording[identifier_field] = identifier
        persisted.append(persisted_recording)
    if len(identifiers) != len(set(identifiers)):
        raise XenoCantoApiError(f"{context} contains duplicate recording identifiers")
    return persisted


def _normalise_api_page(payload: Any, scientific_name: str, requested_page: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise XenoCantoApiError("API response is not a JSON object")
    required = {"numRecordings", "numSpecies", "numPages", "page", "recordings"}
    missing = sorted(required - set(payload))
    if missing:
        raise XenoCantoApiError(f"API response is missing required page fields: {missing}")
    try:
        num_recordings = _strict_positive_int(payload["numRecordings"], "numRecordings")
        num_species = _strict_positive_int(payload["numSpecies"], "numSpecies")
        num_pages = _strict_positive_int(payload["numPages"], "numPages")
        page = _strict_positive_int(payload["page"], "page")
    except ValueError as exc:
        raise XenoCantoApiError(str(exc)) from None
    if num_species != 1:
        raise XenoCantoApiError("exact species query must report exactly one species")
    if page != requested_page:
        raise XenoCantoApiError("API response page does not match the requested page")
    if num_pages > num_recordings or page > num_pages:
        raise XenoCantoApiError("API response page and count relationship is invalid")
    recordings = _canonical_recordings(
        payload["recordings"], scientific_name, f"{scientific_name} page {page}"
    )
    if len(recordings) > num_recordings:
        raise XenoCantoApiError("API page contains more recordings than numRecordings")
    return {
        "page": page,
        "num_recordings": num_recordings,
        "num_species": num_species,
        "num_pages": num_pages,
        "recordings": recordings,
    }


def _redact_message(message: str, secret: str) -> str:
    redacted = message
    for value in _secret_variants(secret):
        redacted = redacted.replace(value, "[redacted]")
    if _contains_request_secret(redacted, secret):
        return "credential-bearing diagnostic removed"
    return _URL_PATTERN.sub("[redacted-url]", redacted)


def format_unknown_acquisition_error(
    error: BaseException,
    *,
    environ: Mapping[str, str] | None = None,
    api_key_environment: str = "XENO_CANTO_API_KEY",
) -> str:
    """Return a user-facing acquisition error without credential or URL disclosure."""
    environment = os.environ if environ is None else environ
    api_key = str(environment.get(api_key_environment) or "").strip()
    return _redact_message(str(error), api_key)


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        _request: urllib.request.Request,
        _file_pointer: Any,
        _code: int,
        _message: str,
        _headers: Any,
        _new_url: str,
    ) -> urllib.request.Request | None:
        raise XenoCantoApiError("API redirects are not permitted")


def _require_approved_request_url(url: str) -> None:
    actual = urllib.parse.urlsplit(url)
    approved = urllib.parse.urlsplit(DEFAULT_ENDPOINT)
    if (
        actual.scheme != "https"
        or actual.scheme != approved.scheme
        or actual.netloc != approved.netloc
        or actual.path != approved.path
        or actual.username is not None
        or actual.password is not None
        or actual.fragment
    ):
        raise XenoCantoApiError("API request target is not the approved HTTPS origin and path")


class UnknownSpeciesApiClient:
    """Sequential API v3 species-page client with redacted retry failures."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: float = 30.0,
        maximum_retries: int = 5,
        request_interval_seconds: float = 1.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("a non-empty XENO_CANTO_API_KEY is required")
        if endpoint != DEFAULT_ENDPOINT:
            raise ValueError("only the approved HTTPS Xeno-canto API v3 endpoint is permitted")
        self.timeout_seconds = _strict_positive_float(timeout_seconds, "timeout_seconds")
        self.maximum_retries = _strict_config_nonnegative_int(maximum_retries, "maximum_retries")
        self.request_interval_seconds = _strict_positive_float(
            request_interval_seconds, "request_interval_seconds"
        )
        if self.request_interval_seconds < 1.0:
            raise ValueError("request_interval_seconds must be at least one second")
        if self.maximum_retries > 10:
            raise ValueError("maximum_retries cannot exceed 10")
        self.endpoint = DEFAULT_ENDPOINT
        self._api_key = api_key.strip()
        self._last_request_started: float | None = None
        self._opener = urllib.request.build_opener(_RejectRedirectHandler())

    def _pace_request(self) -> None:
        now = time.monotonic()
        if self._last_request_started is not None:
            remaining = self.request_interval_seconds - (now - self._last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()

    def _request(self, scientific_name: str, page: int) -> Any:
        query = urllib.parse.urlencode(
            {
                "query": species_query(scientific_name),
                "page": str(page),
                "key": self._api_key,
            },
            quote_via=urllib.parse.quote,
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{query}",
            headers={"User-Agent": "STW7088CEM-bird-audio-coursework/0.1"},
        )
        _require_approved_request_url(request.full_url)
        with self._opener.open(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_page(self, scientific_name: str, page: int) -> dict[str, Any]:
        species_query(scientific_name)
        requested_page = _strict_positive_int(page, "page")
        last_error = "request did not run"
        for attempt in range(self.maximum_retries + 1):
            retry_delay: float | None = None
            try:
                self._pace_request()
                payload = self._request(scientific_name, requested_page)
                if _contains_request_secret(payload, self._api_key):
                    raise XenoCantoApiError("API response contained the request secret")
                if not isinstance(payload, dict):
                    raise XenoCantoApiError("API response is not a JSON object")
                if payload.get("error") not in (None, "", False, [], {}):
                    raise XenoCantoFatalApiError(
                        "API response reported a top-level authentication or query error"
                    )
                return _normalise_api_page(payload, scientific_name, requested_page)
            except urllib.error.HTTPError as exc:
                last_error = _clean_error(exc)
                if exc.code in {400, 401, 403}:
                    raise XenoCantoFatalApiError(
                        f"{scientific_name} page {requested_page}: {last_error}"
                    ) from None
                if exc.code != 429 and not 500 <= exc.code <= 599:
                    raise XenoCantoApiError(
                        f"{scientific_name} page {requested_page}: {last_error}"
                    ) from None
                retry_delay = _retry_after_seconds(exc)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = _clean_error(exc)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise XenoCantoApiError(
                    f"{scientific_name} page {requested_page}: "
                    "API response was not valid UTF-8 JSON"
                ) from None
            except XenoCantoFatalApiError as exc:
                message = _redact_message(str(exc), self._api_key)
                raise XenoCantoFatalApiError(
                    f"{scientific_name} page {requested_page}: {message}"
                ) from None
            except XenoCantoApiError as exc:
                message = _redact_message(str(exc), self._api_key)
                raise XenoCantoApiError(
                    f"{scientific_name} page {requested_page}: {message}"
                ) from None
            except Exception as exc:
                diagnostic = f"{type(exc).__name__}: request failed"
                raise XenoCantoApiError(
                    f"{scientific_name} page {requested_page}: {diagnostic}"
                ) from None
            if attempt < self.maximum_retries:
                time.sleep(retry_delay if retry_delay is not None else min(2**attempt, 16))
        raise XenoCantoApiError(
            f"{scientific_name} page {requested_page}: {_redact_message(last_error, self._api_key)}"
        )


def _new_working_cache(
    config_path: Path,
    config: dict[str, Any],
    config_sha256: str,
    known_species: dict[str, Any],
) -> dict[str, Any]:
    timestamp = _utc_now()
    candidate_pool_target = config["candidate_pool_target_recordings_per_species"]
    recording_target = config["target_recordings_per_species"]
    species_entries: dict[str, Any] = {}
    for species in config["species"]:
        scientific_name = species["scientific_name"]
        species_entries[scientific_name] = {
            **copy.deepcopy(species),
            "candidate_pool_target_recordings": candidate_pool_target,
            "target_recordings": recording_target,
            "query": species_query(scientific_name),
            "snapshot": None,
            "pages": {},
        }
    return {
        "schema_version": UNKNOWN_METADATA_CACHE_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "endpoint": DEFAULT_ENDPOINT,
        "query_form": SPECIES_QUERY_FORM,
        "config_path": _project_relative(config_path),
        "config_sha256": config_sha256,
        "known_species_config_path": _project_relative(known_species["path"]),
        "known_species_config_sha256": known_species["sha256"],
        "known_species_set_sha256": known_species["set_sha256"],
        "known_species_count": known_species["count"],
        "candidate_pool_target_recordings_per_species": candidate_pool_target,
        "target_recordings_per_species": recording_target,
        "created_at_utc": timestamp,
        "updated_at_utc": timestamp,
        "completed_at_utc": None,
        "complete": False,
        "species": species_entries,
    }


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnknownMetadataCacheError(f"JSON artifact is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise UnknownMetadataCacheError(f"JSON artifact is not an object: {path}")
    return value, digest


def _create_json_exclusive(path: str | Path, value: Any) -> Path:
    """Publish complete JSON bytes atomically without replacing an existing path."""
    destination = require_safe_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=False,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validate_cached_recording(recording: Any, scientific_name: str, context: str) -> str:
    if not isinstance(recording, dict):
        raise UnknownMetadataCacheError(f"{context} is not an object")
    unexpected = set(recording) - PERSISTED_RECORDING_FIELDS
    if unexpected:
        raise UnknownMetadataCacheError(f"{context} has unapproved fields: {sorted(unexpected)}")
    try:
        identifier = _validate_recording_identity(recording, scientific_name, context)
    except XenoCantoApiError as exc:
        raise UnknownMetadataCacheError(str(exc)) from None
    for identifier_field in ("id", "nr"):
        if identifier_field in recording and recording[identifier_field] != identifier:
            raise UnknownMetadataCacheError(
                f"{context} identifier is not stored in canonical decimal form"
            )
    return identifier


def _validate_cache(
    cache: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
    config_sha256: str,
    known_species: dict[str, Any],
    *,
    require_complete: bool,
    require_sealed: bool = False,
) -> None:
    expected_keys = _SEALED_CACHE_KEYS if require_sealed else _WORKING_CACHE_KEYS
    _require_exact_keys(cache, expected_keys, "unknown metadata cache", UnknownMetadataCacheError)
    if cache.get("schema_version") != UNKNOWN_METADATA_CACHE_SCHEMA_VERSION:
        raise UnknownMetadataCacheError("unknown metadata cache schema is not supported")
    if cache.get("api_version") != API_VERSION or cache.get("endpoint") != DEFAULT_ENDPOINT:
        raise UnknownMetadataCacheError("unknown metadata cache API contract is invalid")
    if cache.get("query_form") != SPECIES_QUERY_FORM:
        raise UnknownMetadataCacheError("unknown metadata cache query form is invalid")
    if cache.get("config_path") != _project_relative(config_path):
        raise UnknownMetadataCacheError("unknown metadata cache points to a different config")
    if cache.get("config_sha256") != config_sha256:
        raise UnknownMetadataCacheError("unknown metadata cache config hash is invalid")
    expected_known_binding = {
        "known_species_config_path": _project_relative(known_species["path"]),
        "known_species_config_sha256": known_species["sha256"],
        "known_species_set_sha256": known_species["set_sha256"],
        "known_species_count": known_species["count"],
    }
    if any(cache.get(key) != value for key, value in expected_known_binding.items()):
        raise UnknownMetadataCacheError("unknown metadata cache known-species binding is invalid")
    if (
        cache.get("candidate_pool_target_recordings_per_species")
        != CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES
        or cache.get("target_recordings_per_species") != TARGET_RECORDINGS_PER_SPECIES
    ):
        raise UnknownMetadataCacheError("unknown metadata cache recording targets are invalid")
    for timestamp_field in ("created_at_utc", "updated_at_utc"):
        if not str(cache.get(timestamp_field) or "").strip():
            raise UnknownMetadataCacheError(f"unknown metadata cache {timestamp_field} is invalid")
    if not isinstance(cache.get("complete"), bool):
        raise UnknownMetadataCacheError("unknown metadata cache complete flag is invalid")
    if cache["complete"]:
        if not str(cache.get("completed_at_utc") or "").strip():
            raise UnknownMetadataCacheError("completed cache is missing completed_at_utc")
    elif cache.get("completed_at_utc") is not None:
        raise UnknownMetadataCacheError("incomplete cache cannot have completed_at_utc")
    if require_complete and cache.get("complete") is not True:
        raise UnknownMetadataCacheError("unknown metadata cache is incomplete")
    if require_sealed:
        if cache.get("sealed") is not True:
            raise UnknownMetadataCacheError("unknown metadata cache is not sealed")
        if not str(cache.get("sealed_at_utc") or "").strip():
            raise UnknownMetadataCacheError("sealed cache is missing sealed_at_utc")
        if not _SHA256_PATTERN.fullmatch(str(cache.get("source_working_cache_sha256") or "")):
            raise UnknownMetadataCacheError("sealed cache working-cache hash is invalid")
        working_relative = str(cache.get("source_working_cache_path") or "")
        if (
            not working_relative
            or Path(working_relative).is_absolute()
            or not is_relative_to(resolve_project_path(working_relative), PROJECT_ROOT)
        ):
            raise UnknownMetadataCacheError("sealed cache working-cache path is invalid")
        reconstructed_working = {
            key: value for key, value in cache.items() if key not in _SEAL_ONLY_FIELDS
        }
        if sha256_bytes(_canonical_json_bytes(reconstructed_working)) != cache.get(
            "source_working_cache_sha256"
        ):
            raise UnknownMetadataCacheError(
                "sealed cache source working-cache hash is not reproducible"
            )

    entries = cache.get("species")
    if not isinstance(entries, dict):
        raise UnknownMetadataCacheError("unknown metadata cache species field is invalid")
    expected_names = [species["scientific_name"] for species in config["species"]]
    if set(entries) != set(expected_names):
        raise UnknownMetadataCacheError("unknown metadata cache species set is invalid")

    all_recording_ids: set[str] = set()
    for configured_species in config["species"]:
        scientific_name = configured_species["scientific_name"]
        entry = entries[scientific_name]
        if not isinstance(entry, dict):
            raise UnknownMetadataCacheError(f"cache entry for {scientific_name} is invalid")
        _require_exact_keys(
            entry,
            _CACHE_SPECIES_KEYS,
            f"cache entry for {scientific_name}",
            UnknownMetadataCacheError,
        )
        expected_identity = {
            **configured_species,
            "candidate_pool_target_recordings": CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES,
            "target_recordings": TARGET_RECORDINGS_PER_SPECIES,
            "query": species_query(scientific_name),
        }
        if any(entry.get(key) != value for key, value in expected_identity.items()):
            raise UnknownMetadataCacheError(f"cache identity for {scientific_name} is invalid")
        pages = entry.get("pages")
        if not isinstance(pages, dict):
            raise UnknownMetadataCacheError(f"cache pages for {scientific_name} are invalid")
        snapshot = entry.get("snapshot")
        if snapshot is None:
            if pages:
                raise UnknownMetadataCacheError(
                    f"cache pages for {scientific_name} exist without a snapshot"
                )
            if cache["complete"]:
                raise UnknownMetadataCacheError(f"cache snapshot for {scientific_name} is missing")
            continue
        if not isinstance(snapshot, dict):
            raise UnknownMetadataCacheError(f"cache snapshot for {scientific_name} is invalid")
        _require_exact_keys(
            snapshot,
            _SNAPSHOT_KEYS,
            f"cache snapshot for {scientific_name}",
            UnknownMetadataCacheError,
        )
        try:
            num_recordings = _strict_positive_int(
                snapshot["num_recordings"], f"{scientific_name} num_recordings"
            )
            num_species = _strict_positive_int(
                snapshot["num_species"], f"{scientific_name} num_species"
            )
            num_pages = _strict_positive_int(snapshot["num_pages"], f"{scientific_name} num_pages")
        except ValueError as exc:
            raise UnknownMetadataCacheError(str(exc)) from None
        if num_species != 1 or num_pages > num_recordings:
            raise UnknownMetadataCacheError(f"cache snapshot for {scientific_name} is invalid")
        page_numbers: list[int] = []
        try:
            for key in pages:
                if not isinstance(key, str):
                    raise ValueError("cache page key must be a canonical decimal string")
                page_number = _strict_positive_int(key, "cache page key")
                if key != str(page_number):
                    raise ValueError("cache page key must be a canonical decimal string")
                page_numbers.append(page_number)
            page_numbers.sort()
        except ValueError as exc:
            raise UnknownMetadataCacheError(str(exc)) from None
        if len(page_numbers) != len(set(page_numbers)) or page_numbers != list(
            range(1, len(page_numbers) + 1)
        ):
            raise UnknownMetadataCacheError(
                f"cache pages for {scientific_name} are not contiguous from page one"
            )
        if page_numbers and page_numbers[-1] > num_pages:
            raise UnknownMetadataCacheError(f"cache has an out-of-range page for {scientific_name}")

        species_ids: list[str] = []
        for page_number in page_numbers:
            page_entry = pages[str(page_number)]
            if not isinstance(page_entry, dict):
                raise UnknownMetadataCacheError(
                    f"cache page {page_number} for {scientific_name} is invalid"
                )
            _require_exact_keys(
                page_entry,
                _PAGE_KEYS,
                f"cache page {page_number} for {scientific_name}",
                UnknownMetadataCacheError,
            )
            if not str(page_entry.get("fetched_at_utc") or "").strip():
                raise UnknownMetadataCacheError(
                    f"cache page {page_number} for {scientific_name} has no timestamp"
                )
            expected_page_counts = {
                "page": page_number,
                "num_recordings": num_recordings,
                "num_species": num_species,
                "num_pages": num_pages,
            }
            if any(page_entry.get(key) != value for key, value in expected_page_counts.items()):
                raise UnknownMetadataCacheError(
                    f"cache page/count snapshot drift at {scientific_name} page {page_number}"
                )
            recordings = page_entry.get("recordings")
            if not isinstance(recordings, list) or not recordings:
                raise UnknownMetadataCacheError(
                    f"cache page {page_number} for {scientific_name} has no recordings"
                )
            identifiers = [
                _validate_cached_recording(
                    recording,
                    scientific_name,
                    f"{scientific_name} page {page_number} recording[{index}]",
                )
                for index, recording in enumerate(recordings)
            ]
            if len(identifiers) != len(set(identifiers)):
                raise UnknownMetadataCacheError(
                    f"cache page {page_number} for {scientific_name} has duplicates"
                )
            if page_entry.get("recording_count") != len(recordings):
                raise UnknownMetadataCacheError(
                    f"cache page {page_number} for {scientific_name} count is invalid"
                )
            if page_entry.get("recording_ids_sha256") != sha256_json(identifiers):
                raise UnknownMetadataCacheError(
                    f"cache page {page_number} for {scientific_name} ID hash is invalid"
                )
            species_ids.extend(identifiers)
        if len(species_ids) != len(set(species_ids)):
            raise UnknownMetadataCacheError(
                f"cache contains cross-page duplicates for {scientific_name}"
            )
        overlap = all_recording_ids.intersection(species_ids)
        if overlap:
            raise UnknownMetadataCacheError("cache recording identifiers overlap across species")
        all_recording_ids.update(species_ids)
        if cache["complete"]:
            if page_numbers != list(range(1, num_pages + 1)):
                raise UnknownMetadataCacheError(f"cache pages for {scientific_name} are incomplete")
            if len(species_ids) != num_recordings:
                raise UnknownMetadataCacheError(
                    f"cache recording total for {scientific_name} does not match numRecordings"
                )


def _validate_fetched_page(
    result: Any, scientific_name: str, requested_page: int, api_key: str
) -> dict[str, Any]:
    if _contains_request_secret(result, api_key):
        raise XenoCantoApiError("API page result contained the request secret")
    if not isinstance(result, dict):
        raise XenoCantoApiError("canonical API page result is not an object")
    _require_exact_keys(result, _CANONICAL_PAGE_KEYS, "canonical API page", XenoCantoApiError)
    try:
        page = _strict_positive_int(result["page"], "page")
        num_recordings = _strict_positive_int(result["num_recordings"], "num_recordings")
        num_species = _strict_positive_int(result["num_species"], "num_species")
        num_pages = _strict_positive_int(result["num_pages"], "num_pages")
    except ValueError as exc:
        raise XenoCantoApiError(str(exc)) from None
    if page != requested_page or page > num_pages or num_species != 1 or num_pages > num_recordings:
        raise XenoCantoApiError("canonical API page count or page identity is invalid")
    recordings = _canonical_recordings(
        result["recordings"], scientific_name, f"{scientific_name} page {page}"
    )
    if len(recordings) > num_recordings:
        raise XenoCantoApiError("canonical API page exceeds num_recordings")
    return {
        "page": page,
        "num_recordings": num_recordings,
        "num_species": num_species,
        "num_pages": num_pages,
        "recordings": recordings,
    }


def _page_entry(page: dict[str, Any]) -> dict[str, Any]:
    identifiers = [
        _strict_recording_id(recording, "API page recording") for recording in page["recordings"]
    ]
    return {
        "fetched_at_utc": _utc_now(),
        "page": page["page"],
        "num_recordings": page["num_recordings"],
        "num_species": page["num_species"],
        "num_pages": page["num_pages"],
        "recording_count": len(page["recordings"]),
        "recording_ids_sha256": sha256_json(identifiers),
        "recordings": page["recordings"],
    }


def _checkpoint_cache(
    destination: Path,
    cache: dict[str, Any],
    expected_cache_sha256: str | None,
    config_path: Path,
    config_sha256: str,
    known_species: dict[str, Any],
) -> str:
    require_unchanged(config_path, config_sha256)
    require_unchanged(known_species["path"], known_species["sha256"])
    if expected_cache_sha256 is None:
        if destination.exists():
            raise RuntimeError(f"Working cache appeared during command execution: {destination}")
    else:
        require_unchanged(destination, expected_cache_sha256)
    cache["updated_at_utc"] = _utc_now()
    atomic_write_json(destination, cache)
    return sha256_file(destination)


def _canonical_cached_page(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": page["page"],
        "num_recordings": page["num_recordings"],
        "num_species": page["num_species"],
        "num_pages": page["num_pages"],
        "recordings": page["recordings"],
    }


def _emit_progress(
    callback: ProgressCallback | None,
    phase: str,
    scientific_name: str,
    page: int,
    total_pages: int,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "scientific_name": scientific_name,
                "page": page,
                "total_pages": total_pages,
            }
        )


def _snapshot_drift_error(scientific_name: str, page_number: int, detail: str) -> XenoCantoApiError:
    return XenoCantoApiError(
        f"{scientific_name} page {page_number} {detail}. "
        "Retain this working cache as evidence and restart discovery with a fresh "
        "--working-cache path"
    )


def _fetch_validated_page(
    client: UnknownSpeciesApiClient,
    scientific_name: str,
    page_number: int,
    api_key: str,
    phase: str,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    fetched = client.fetch_page(scientific_name, page_number)
    page = _validate_fetched_page(fetched, scientific_name, page_number, api_key)
    _emit_progress(
        progress_callback,
        phase,
        scientific_name,
        page["page"],
        page["num_pages"],
    )
    return page


def _revalidate_cached_pages(
    client: UnknownSpeciesApiClient,
    cache: dict[str, Any],
    config: dict[str, Any],
    api_key: str,
    phase: str,
    progress_callback: ProgressCallback | None,
) -> None:
    for configured_species in config["species"]:
        scientific_name = configured_species["scientific_name"]
        entry = cache["species"][scientific_name]
        snapshot = entry["snapshot"]
        if snapshot is None:
            continue
        for page_number in range(1, len(entry["pages"]) + 1):
            page = _fetch_validated_page(
                client,
                scientific_name,
                page_number,
                api_key,
                phase,
                progress_callback,
            )
            expected_snapshot = {
                "num_recordings": page["num_recordings"],
                "num_species": page["num_species"],
                "num_pages": page["num_pages"],
            }
            if expected_snapshot != snapshot:
                raise _snapshot_drift_error(
                    scientific_name,
                    page_number,
                    "changed the count/page snapshot",
                )
            if page != _canonical_cached_page(entry["pages"][str(page_number)]):
                raise _snapshot_drift_error(
                    scientific_name,
                    page_number,
                    "changed persisted page content",
                )


def fetch_unknown_metadata_cache(
    config_path: str | Path,
    working_cache_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fetch every server-reported metadata page into a resumable working cache."""
    destination = require_safe_output(working_cache_path)
    config_file = _require_project_path(config_path, "unknown acquisition config")
    _require_project_path(KNOWN_SPECIES_CONFIG_PATH, "known-species config")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    environment = os.environ if environ is None else environ
    with project_lock("unknown_metadata_discovery"):
        config_file, config, config_sha256 = _load_config_snapshot(config_file)
        known_species = _load_known_species_snapshot(config)
        if destination.exists():
            cache, cache_sha256 = _read_json_snapshot(destination)
            _validate_cache(
                cache,
                config_file,
                config,
                config_sha256,
                known_species,
                require_complete=False,
            )
        else:
            cache = _new_working_cache(config_file, config, config_sha256, known_species)
            cache_sha256 = None

        if cache.get("complete") is True:
            return destination, cache

        api = config["api"]
        environment_name = api["api_key_environment"]
        api_key = str(environment.get(environment_name) or "").strip()
        if not api_key:
            raise UnknownAcquisitionCredentialError(
                f"a non-empty {environment_name} environment variable is required"
            )
        client = UnknownSpeciesApiClient(
            api_key,
            endpoint=api["endpoint"],
            timeout_seconds=api["timeout_seconds"],
            maximum_retries=api["maximum_retries"],
            request_interval_seconds=api["request_interval_seconds"],
        )

        if cache_sha256 is None:
            cache_sha256 = _checkpoint_cache(
                destination,
                cache,
                cache_sha256,
                config_file,
                config_sha256,
                known_species,
            )

        _revalidate_cached_pages(
            client,
            cache,
            config,
            api_key,
            "resume_revalidation",
            progress_callback,
        )

        for configured_species in config["species"]:
            scientific_name = configured_species["scientific_name"]
            cache_species = cache["species"][scientific_name]
            snapshot = cache_species["snapshot"]
            next_page = len(cache_species["pages"]) + 1
            if snapshot is None:
                next_page = 1
                last_page = 1
            else:
                last_page = snapshot["num_pages"]

            while next_page <= last_page:
                page = _fetch_validated_page(
                    client,
                    scientific_name,
                    next_page,
                    api_key,
                    "fetch",
                    progress_callback,
                )
                page_snapshot = {
                    "num_recordings": page["num_recordings"],
                    "num_species": page["num_species"],
                    "num_pages": page["num_pages"],
                }
                if snapshot is None:
                    snapshot = page_snapshot
                    cache_species["snapshot"] = snapshot
                    last_page = snapshot["num_pages"]
                elif page_snapshot != snapshot:
                    raise _snapshot_drift_error(
                        scientific_name,
                        next_page,
                        "changed the count/page snapshot",
                    )

                existing_ids = {
                    _strict_recording_id(recording, "cached recording")
                    for existing_species in cache["species"].values()
                    for existing_page in existing_species["pages"].values()
                    for recording in existing_page["recordings"]
                }
                new_ids = {
                    _strict_recording_id(recording, "API page recording")
                    for recording in page["recordings"]
                }
                if existing_ids.intersection(new_ids):
                    raise XenoCantoApiError(
                        f"{scientific_name} page {next_page} duplicates a prior cached recording"
                    )
                cache_species["pages"][str(next_page)] = _page_entry(page)
                _validate_cache(
                    cache,
                    config_file,
                    config,
                    config_sha256,
                    known_species,
                    require_complete=False,
                )
                cache_sha256 = _checkpoint_cache(
                    destination,
                    cache,
                    cache_sha256,
                    config_file,
                    config_sha256,
                    known_species,
                )
                next_page += 1

        _revalidate_cached_pages(
            client,
            cache,
            config,
            api_key,
            "completion_revalidation",
            progress_callback,
        )
        cache["complete"] = True
        cache["completed_at_utc"] = _utc_now()
        _validate_cache(
            cache,
            config_file,
            config,
            config_sha256,
            known_species,
            require_complete=True,
        )
        _checkpoint_cache(
            destination,
            cache,
            cache_sha256,
            config_file,
            config_sha256,
            known_species,
        )
    return destination, cache


def _cache_summary(cache: dict[str, Any]) -> dict[str, Any]:
    species_counts: dict[str, int] = {}
    recording_keys: list[str] = []
    snapshots: list[dict[str, Any]] = []
    for scientific_name, entry in cache["species"].items():
        snapshot = entry["snapshot"]
        species_counts[scientific_name] = snapshot["num_recordings"]
        snapshots.append(
            {
                "scientific_name": scientific_name,
                "num_recordings": snapshot["num_recordings"],
                "num_species": snapshot["num_species"],
                "num_pages": snapshot["num_pages"],
            }
        )
        for page in entry["pages"].values():
            for recording in page["recordings"]:
                recording_keys.append(
                    f"{scientific_name}:XC{_strict_recording_id(recording, 'sealed recording')}"
                )
    return {
        "species_recording_counts": species_counts,
        "recordings_total": sum(species_counts.values()),
        "recording_set_sha256": sha256_json(sorted(recording_keys)),
        "species_snapshot_sha256": sha256_json(
            sorted(snapshots, key=lambda item: item["scientific_name"])
        ),
    }


def seal_unknown_metadata_cache(
    config_path: str | Path,
    working_cache_path: str | Path,
    output_path: str | Path,
    lock_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Seal a complete discovery cache and write its hash-bound planning lock."""
    config_file = _require_project_path(config_path, "unknown acquisition config")
    working_cache = _require_project_path(working_cache_path, "working cache")
    _require_project_path(KNOWN_SPECIES_CONFIG_PATH, "known-species config")
    destination = require_safe_output(output_path)
    lock_destination = require_safe_output(lock_path)
    if len({config_file, working_cache, destination, lock_destination}) != 4:
        raise ValueError("config, working cache, sealed cache, and lock paths must be distinct")

    with project_lock("unknown_metadata_discovery"):
        if lock_destination.exists():
            if not destination.exists():
                raise FileExistsError("Unknown metadata lock exists without its sealed cache")
            verified = verify_unknown_metadata_lock(lock_destination, destination)
            artifacts = verified["artifacts"]
            if resolve_project_path(artifacts["config"]["path"]) != config_file:
                raise UnknownMetadataCacheError(
                    "existing unknown metadata lock points to a different config"
                )
            if resolve_project_path(artifacts["working_cache"]["path"]) != working_cache:
                raise UnknownMetadataCacheError(
                    "existing unknown metadata lock points to a different working cache"
                )
            return destination, lock_destination, verified

        config_file, config, config_sha256 = _load_config_snapshot(config_file)
        known_species = _load_known_species_snapshot(config)
        if destination.exists():
            sealed_cache, sealed_cache_sha256 = _read_json_snapshot(destination)
            _validate_cache(
                sealed_cache,
                config_file,
                config,
                config_sha256,
                known_species,
                require_complete=True,
                require_sealed=True,
            )
            if resolve_project_path(sealed_cache["source_working_cache_path"]) != working_cache:
                raise UnknownMetadataCacheError(
                    "existing sealed cache points to a different working cache"
                )
            working_cache_sha256 = sealed_cache["source_working_cache_sha256"]
        else:
            cache, working_cache_sha256 = _read_json_snapshot(working_cache)
            _validate_cache(
                cache,
                config_file,
                config,
                config_sha256,
                known_species,
                require_complete=True,
            )
            sealed_cache = copy.deepcopy(cache)
            sealed_cache.update(
                {
                    "sealed": True,
                    "sealed_at_utc": _utc_now(),
                    "source_working_cache_path": _project_relative(working_cache),
                    "source_working_cache_sha256": working_cache_sha256,
                }
            )
            _validate_cache(
                sealed_cache,
                config_file,
                config,
                config_sha256,
                known_species,
                require_complete=True,
                require_sealed=True,
            )
            require_unchanged(config_file, config_sha256)
            require_unchanged(known_species["path"], known_species["sha256"])
            require_unchanged(working_cache, working_cache_sha256)
            _create_json_exclusive(destination, sealed_cache)
            sealed_cache_sha256 = sha256_file(destination)

        summary = _cache_summary(sealed_cache)
        lock = {
            "schema_version": UNKNOWN_METADATA_LOCK_SCHEMA_VERSION,
            "locked_at_utc": _utc_now(),
            "ready_for_candidate_planning": True,
            "api_version": API_VERSION,
            "endpoint": DEFAULT_ENDPOINT,
            "query_form": SPECIES_QUERY_FORM,
            "pagination_policy": config["api"]["pagination_policy"],
            "snapshot_consistency": config["api"]["snapshot_consistency"],
            "species_count": len(config["species"]),
            "primary_species_count": sum(
                species["role"] == "primary" for species in config["species"]
            ),
            "fallback_species_count": sum(
                species["role"] == "fallback" for species in config["species"]
            ),
            "inactive_fallback_count": sum(
                species["role"] == "fallback" and not species["active"]
                for species in config["species"]
            ),
            "candidate_pool_target_recordings_per_species": (
                CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES
            ),
            "target_recordings_per_species": TARGET_RECORDINGS_PER_SPECIES,
            "config_sha256": config_sha256,
            "known_species_config_sha256": known_species["sha256"],
            "known_species_set_sha256": known_species["set_sha256"],
            "known_species_count": known_species["count"],
            "source_working_cache_sha256": working_cache_sha256,
            "sealed_cache_sha256": sealed_cache_sha256,
            **summary,
            "artifacts": {
                "config": {
                    "path": _project_relative(config_file),
                    "sha256": config_sha256,
                },
                "working_cache": {
                    "path": _project_relative(working_cache),
                    "sha256": working_cache_sha256,
                },
                "known_species_config": {
                    "path": _project_relative(known_species["path"]),
                    "sha256": known_species["sha256"],
                },
                "sealed_cache": {
                    "path": _project_relative(destination),
                    "sha256": sealed_cache_sha256,
                },
            },
        }
        require_unchanged(config_file, config_sha256)
        require_unchanged(known_species["path"], known_species["sha256"])
        require_unchanged(destination, sealed_cache_sha256)
        _create_json_exclusive(lock_destination, lock)
    return destination, lock_destination, lock


def _verify_artifacts(lock: dict[str, Any], required: set[str]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise UnknownMetadataCacheError("unknown metadata lock is missing artifacts")
    resolved: dict[str, Path] = {}
    for name in required:
        entry = artifacts[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise UnknownMetadataCacheError(f"unknown metadata lock artifact is invalid: {name}")
        relative = str(entry.get("path") or "")
        expected_sha256 = str(entry.get("sha256") or "")
        if (
            not relative
            or Path(relative).is_absolute()
            or not _SHA256_PATTERN.fullmatch(expected_sha256)
        ):
            raise UnknownMetadataCacheError(f"unknown metadata lock artifact is invalid: {name}")
        path = resolve_project_path(relative)
        if not is_relative_to(path, PROJECT_ROOT):
            raise UnknownMetadataCacheError(f"unknown metadata lock path leaves project: {name}")
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise UnknownMetadataCacheError(f"unknown metadata lock hash check failed: {name}")
        resolved[name] = path
    return resolved


def verify_unknown_metadata_lock(
    lock_path: str | Path,
    expected_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the immutable config and sealed-cache snapshot bound by a lock."""
    lock_file = _require_project_path(lock_path, "unknown metadata lock")
    expected_cache = (
        _require_project_path(expected_cache_path, "expected sealed cache")
        if expected_cache_path is not None
        else None
    )
    lock, _ = _read_json_snapshot(lock_file)
    _require_exact_keys(lock, _LOCK_KEYS, "unknown metadata lock", UnknownMetadataCacheError)
    if not str(lock.get("locked_at_utc") or "").strip():
        raise UnknownMetadataCacheError("unknown metadata lock timestamp is invalid")
    for hash_field in (
        "config_sha256",
        "known_species_config_sha256",
        "known_species_set_sha256",
        "source_working_cache_sha256",
        "sealed_cache_sha256",
        "recording_set_sha256",
        "species_snapshot_sha256",
    ):
        if not _SHA256_PATTERN.fullmatch(str(lock.get(hash_field) or "")):
            raise UnknownMetadataCacheError(f"unknown metadata lock hash is invalid: {hash_field}")
    if lock.get("schema_version") != UNKNOWN_METADATA_LOCK_SCHEMA_VERSION:
        raise UnknownMetadataCacheError("unknown metadata lock schema is not supported")
    if lock.get("ready_for_candidate_planning") is not True:
        raise UnknownMetadataCacheError("unknown metadata lock is not ready for planning")
    if (
        lock.get("api_version") != API_VERSION
        or lock.get("endpoint") != DEFAULT_ENDPOINT
        or lock.get("query_form") != SPECIES_QUERY_FORM
    ):
        raise UnknownMetadataCacheError("unknown metadata lock API contract is invalid")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "config",
        "known_species_config",
        "working_cache",
        "sealed_cache",
    }:
        raise UnknownMetadataCacheError("unknown metadata lock artifact table is invalid")
    working_entry = artifacts["working_cache"]
    if not isinstance(working_entry, dict) or set(working_entry) != {"path", "sha256"}:
        raise UnknownMetadataCacheError("unknown metadata working-cache reference is invalid")
    working_relative = str(working_entry.get("path") or "")
    working_sha256 = str(working_entry.get("sha256") or "")
    if (
        not working_relative
        or Path(working_relative).is_absolute()
        or not is_relative_to(resolve_project_path(working_relative), PROJECT_ROOT)
        or not _SHA256_PATTERN.fullmatch(working_sha256)
        or working_sha256 != lock.get("source_working_cache_sha256")
    ):
        raise UnknownMetadataCacheError("unknown metadata working-cache reference is invalid")
    paths = _verify_artifacts(lock, {"config", "known_species_config", "sealed_cache"})
    if expected_cache is not None and paths["sealed_cache"] != expected_cache:
        raise UnknownMetadataCacheError("unknown metadata lock points to a different cache")
    config_path, config, config_sha256 = _load_config_snapshot(paths["config"])
    known_species = _load_known_species_snapshot(config, paths["known_species_config"])
    sealed_cache, sealed_cache_sha256 = _read_json_snapshot(paths["sealed_cache"])
    _validate_cache(
        sealed_cache,
        config_path,
        config,
        config_sha256,
        known_species,
        require_complete=True,
        require_sealed=True,
    )
    if working_relative != sealed_cache.get("source_working_cache_path"):
        raise UnknownMetadataCacheError("unknown metadata lock working-cache path is inconsistent")
    if lock.get("config_sha256") != config_sha256:
        raise UnknownMetadataCacheError("unknown metadata lock config hash is inconsistent")
    if (
        lock.get("known_species_config_sha256") != known_species["sha256"]
        or lock.get("known_species_set_sha256") != known_species["set_sha256"]
        or lock.get("known_species_count") != known_species["count"]
    ):
        raise UnknownMetadataCacheError(
            "unknown metadata lock known-species binding is inconsistent"
        )
    if lock.get("sealed_cache_sha256") != sealed_cache_sha256:
        raise UnknownMetadataCacheError("unknown metadata lock cache hash is inconsistent")
    if lock.get("source_working_cache_sha256") != sealed_cache.get("source_working_cache_sha256"):
        raise UnknownMetadataCacheError("unknown metadata lock working hash is inconsistent")
    expected_scalars = {
        "pagination_policy": "all_server_reported_pages_in_ascending_order",
        "snapshot_consistency": "exact_numRecordings_numSpecies_numPages_and_page",
        "species_count": 6,
        "primary_species_count": 5,
        "fallback_species_count": 1,
        "inactive_fallback_count": 1,
        "known_species_count": 15,
        "candidate_pool_target_recordings_per_species": (
            CANDIDATE_POOL_TARGET_RECORDINGS_PER_SPECIES
        ),
        "target_recordings_per_species": TARGET_RECORDINGS_PER_SPECIES,
    }
    if any(lock.get(key) != value for key, value in expected_scalars.items()):
        raise UnknownMetadataCacheError("unknown metadata lock protocol summary is invalid")
    summary = _cache_summary(sealed_cache)
    if any(lock.get(key) != value for key, value in summary.items()):
        raise UnknownMetadataCacheError("unknown metadata lock snapshot summary is invalid")
    return lock
