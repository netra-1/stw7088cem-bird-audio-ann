from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from bird_audio.config import load_toml
from bird_audio.hashing import sha256_bytes, sha256_file, sha256_json
from bird_audio.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_snapshot,
    require_unchanged,
)
from bird_audio.locking import project_lock
from bird_audio.manifest import LOCAL_MANIFEST_FIELDS, apply_qc_reason
from bird_audio.paths import require_safe_output, resolve_project_path

DEFAULT_ENDPOINT = "https://xeno-canto.org/api/3/recordings"
API_VERSION = "xeno-canto_api_v3"
MAXIMUM_RETRY_AFTER_SECONDS = 60.0
ENRICHED_MANIFEST_FIELDS = [
    *LOCAL_MANIFEST_FIELDS,
    "identity_validation_status",
    "licence_validation_status",
]
LICENCE_FIELDS = [
    "recording_id",
    "xc_url",
    "species_common_name",
    "recordist",
    "licence",
    "licence_validation_status",
    "attribution",
    "local_qc_status",
]
SAME_INDIVIDUAL_PATTERN = re.compile(
    r"\b(?:same\s+(?:bird|individual|specimen)|same\s+as|"
    r"identical\s+(?:bird|individual|specimen))\b",
    flags=re.IGNORECASE,
)
XC_REFERENCE_PATTERN = re.compile(r"\bXC\s?(\d{3,})\b", flags=re.IGNORECASE)
PERSISTED_RECORDING_FIELDS = frozenset(
    {
        "id",
        "nr",
        "gen",
        "sp",
        "ssp",
        "en",
        "grp",
        "group",
        "also",
        "rec",
        "cnt",
        "loc",
        "lat",
        "lon",
        "lng",
        "date",
        "time",
        "q",
        "type",
        "sex",
        "stage",
        "rmk",
        "dvc",
        "mic",
        "smp",
        "lic",
        "file",
        "file-name",
        "length",
        "method",
        "playback-used",
        "auto",
        "uploaded",
        "url",
    }
)


class XenoCantoApiError(RuntimeError):
    pass


class XenoCantoFatalApiError(XenoCantoApiError):
    """An API failure that would invalidate every remaining per-recording request."""


class XenoCantoRecordUnavailableError(XenoCantoApiError):
    """A terminal absence for one requested recording, safe to exclude and continue."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _clean_error(exc: BaseException) -> str:
    """Return an error description that cannot include a request URL or API key."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError status={exc.code}"
    if isinstance(exc, urllib.error.URLError):
        reason = type(exc.reason).__name__ if exc.reason is not None else "unknown"
        return f"URLError reason={reason}"
    if isinstance(exc, TimeoutError):
        return "TimeoutError: request timed out"
    return f"{type(exc).__name__}: request failed"


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """Return a standards-aware Retry-After delay with a conservative upper bound."""
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    if value is None:
        return None
    text = str(value).strip()
    try:
        delay = float(text)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(text)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = (retry_at.astimezone(UTC) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(delay):
        return None
    return min(max(delay, 0.0), MAXIMUM_RETRY_AFTER_SECONDS)


def _recording_id(recording: dict[str, Any]) -> str:
    value = str(recording.get("id") or recording.get("nr") or "")
    return value.removeprefix("XC")


def _contains_secret(value: Any, secret: str) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_secret(key, secret) or _contains_secret(item, secret)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item, secret) for item in value)
    return bool(secret and secret in str(value))


class XenoCantoClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: float = 30,
        maximum_retries: int = 5,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A non-empty XENO_CANTO_API_KEY is required")
        if endpoint.rstrip("/") != DEFAULT_ENDPOINT:
            raise ValueError("Only the approved HTTPS Xeno-canto API v3 endpoint is permitted")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if maximum_retries < 0:
            raise ValueError("maximum_retries cannot be negative")
        self._api_key = api_key.strip()
        self.endpoint = DEFAULT_ENDPOINT
        self.timeout_seconds = timeout_seconds
        self.maximum_retries = maximum_retries

    def _request(self, xc_id: str) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {"query": f"nr:{xc_id}", "key": self._api_key},
            quote_via=urllib.parse.quote,
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{query}",
            headers={"User-Agent": "STW7088CEM-bird-audio-coursework/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise XenoCantoApiError("API response is not a JSON object")
        return payload

    def fetch_recording(self, xc_id: str) -> dict[str, Any]:
        xc_id = str(xc_id).removeprefix("XC")
        if not xc_id.isdigit():
            raise ValueError("xc_id must contain only digits")
        last_error = "request did not run"
        for attempt in range(self.maximum_retries + 1):
            retry_delay: float | None = None
            try:
                payload = self._request(xc_id)
                if _contains_secret(payload, self._api_key):
                    raise XenoCantoApiError("API response contained the request secret")
                if payload.get("error") not in (None, "", False, [], {}):
                    raise XenoCantoFatalApiError(
                        "API response reported a top-level authentication or query error"
                    )
                for count_field in ("numRecordings", "nr"):
                    if count_field not in payload:
                        continue
                    try:
                        result_count = int(str(payload[count_field]).strip())
                    except (TypeError, ValueError):
                        raise XenoCantoApiError(
                            "API response has an invalid top-level result count"
                        ) from None
                    if result_count == 0:
                        raise XenoCantoRecordUnavailableError(
                            "API response reports no matching recording"
                        )
                    if result_count != 1:
                        raise XenoCantoApiError("API response top-level result count is not one")
                recordings = payload.get("recordings")
                if isinstance(recordings, list) and not recordings:
                    raise XenoCantoRecordUnavailableError(
                        "API response contains no matching recording"
                    )
                if not isinstance(recordings, list) or len(recordings) != 1:
                    raise XenoCantoApiError(f"Expected exactly one recording for XC{xc_id}")
                recording = recordings[0]
                if not isinstance(recording, dict) or _recording_id(recording) != xc_id:
                    raise XenoCantoApiError(
                        f"API response recording identity does not match XC{xc_id}"
                    )
                return {
                    key: value
                    for key, value in recording.items()
                    if key in PERSISTED_RECORDING_FIELDS
                }
            except urllib.error.HTTPError as exc:
                last_error = _clean_error(exc)
                if exc.code in {400, 401, 403}:
                    raise XenoCantoFatalApiError(f"XC{xc_id}: {last_error}") from None
                if exc.code == 404:
                    raise XenoCantoRecordUnavailableError(f"XC{xc_id}: {last_error}") from None
                if exc.code != 429 and not 500 <= exc.code <= 599:
                    raise XenoCantoApiError(f"XC{xc_id}: {last_error}") from None
                retry_delay = _retry_after_seconds(exc)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = _clean_error(exc)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise XenoCantoApiError(
                    f"XC{xc_id}: API response was not valid UTF-8 JSON"
                ) from None
            except XenoCantoFatalApiError as exc:
                message = str(exc).replace(self._api_key, "[redacted]")
                raise XenoCantoFatalApiError(f"XC{xc_id}: {message}") from None
            except XenoCantoRecordUnavailableError as exc:
                message = str(exc).replace(self._api_key, "[redacted]")
                raise XenoCantoRecordUnavailableError(f"XC{xc_id}: {message}") from None
            except XenoCantoApiError as exc:
                message = str(exc).replace(self._api_key, "[redacted]")
                raise XenoCantoApiError(f"XC{xc_id}: {message}") from None
            if attempt < self.maximum_retries:
                time.sleep(retry_delay if retry_delay is not None else min(2**attempt, 16))
        raise XenoCantoApiError(f"XC{xc_id}: {last_error}")


def _new_cache(endpoint: str) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "api_version": API_VERSION,
        "endpoint": endpoint,
        "query_form": "nr:<xc_id>",
        "created_at_utc": _utc_now(),
        "records": {},
    }


def _validate_cache_payload(
    payload: Any,
    path: Path,
    expected_endpoint: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metadata cache payload: {path}")
    if payload.get("schema_version") != "1.1" or not isinstance(payload.get("records"), dict):
        raise ValueError(f"Invalid metadata cache schema: {path}")
    if payload.get("api_version") != API_VERSION:
        raise ValueError(f"Metadata cache API version mismatch: {path}")
    if payload.get("endpoint") != expected_endpoint or payload.get("query_form") != "nr:<xc_id>":
        raise ValueError(f"Metadata cache endpoint or query-form mismatch: {path}")
    for xc_id, entry in payload["records"].items():
        if not str(xc_id).isdigit() or not isinstance(entry, dict):
            raise ValueError(f"Invalid metadata cache recording key: {xc_id!r}")
        status = entry.get("status")
        if status not in {"ok", "error", "unavailable"}:
            raise ValueError(f"Invalid metadata cache status for XC{xc_id}")
        recording = entry.get("recording")
        if status == "ok" and (
            not isinstance(recording, dict) or _recording_id(recording) != str(xc_id)
        ):
            raise ValueError(f"Metadata cache identity mismatch for XC{xc_id}")
        if status == "ok" and not set(recording).issubset(PERSISTED_RECORDING_FIELDS):
            raise ValueError(f"Metadata cache contains unapproved fields for XC{xc_id}")
        if status in {"error", "unavailable"} and recording not in ({}, None):
            raise ValueError(f"Failed metadata cache entry contains a payload for XC{xc_id}")
    return payload


def load_metadata_cache_snapshot(
    path: str | Path,
    expected_endpoint: str = DEFAULT_ENDPOINT,
) -> tuple[dict[str, Any], str]:
    """Parse and hash the same immutable metadata-cache byte snapshot."""
    cache_path = Path(path)
    raw = cache_path.read_bytes()
    digest = sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid metadata cache JSON: {cache_path}") from exc
    return _validate_cache_payload(payload, cache_path, expected_endpoint), digest


def _load_cache(path: Path, expected_endpoint: str = DEFAULT_ENDPOINT) -> dict[str, Any]:
    if not path.exists():
        return _new_cache(expected_endpoint)
    return load_metadata_cache_snapshot(path, expected_endpoint)[0]


def fetch_metadata_cache(
    local_manifest_path: str | Path,
    cache_path: str | Path,
    api_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    request_interval_seconds: float = 1.0,
    checkpoint_every: int = 20,
    maximum_retries: int = 5,
    timeout_seconds: float = 30,
) -> tuple[Path, dict[str, Any]]:
    if request_interval_seconds < 1.0:
        raise ValueError("request_interval_seconds must be at least 1.0")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    manifest_path = resolve_project_path(local_manifest_path)
    destination = require_safe_output(cache_path)
    client = XenoCantoClient(
        api_key=api_key,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        maximum_retries=maximum_retries,
    )
    with project_lock("metadata_cache"):
        rows, manifest_sha256 = read_csv_snapshot(manifest_path)
        cache = _load_cache(destination, expected_endpoint=client.endpoint)
        if cache.get("sealed") is True:
            raise ValueError("A sealed metadata cache cannot be reopened for fetching")
        records: dict[str, Any] = cache["records"]
        xc_ids = sorted({row["xc_id"] for row in rows}, key=lambda value: int(value))
        recording_ids_sha256 = sha256_json(xc_ids)
        cached_manifest_hash = cache.get("source_manifest_sha256")
        if records and cached_manifest_hash != manifest_sha256:
            raise ValueError(
                "Non-empty metadata cache is bound to a different exact local manifest"
            )
        cached_id_hash = cache.get("source_recording_ids_sha256")
        if cached_id_hash and cached_id_hash != recording_ids_sha256:
            raise ValueError("Metadata cache recording set does not match the local manifest")
        unexpected_ids = sorted(set(records) - set(xc_ids), key=int)
        if unexpected_ids:
            raise ValueError(
                f"Metadata cache contains unexpected recording IDs: {unexpected_ids[:20]}"
            )
        cache.update(
            {
                "api_version": API_VERSION,
                "endpoint": client.endpoint,
                "query_form": "nr:<xc_id>",
                "source_manifest_sha256": manifest_sha256,
                "source_recording_ids_sha256": recording_ids_sha256,
                "updated_at_utc": _utc_now(),
            }
        )
        pending = [
            xc_id
            for xc_id in xc_ids
            if xc_id not in records or records[xc_id].get("status") == "error"
        ]
        print(f"Metadata cache contains {len(xc_ids) - len(pending)}/{len(xc_ids)} recordings")
        for index, xc_id in enumerate(pending, start=1):
            try:
                recording = client.fetch_recording(xc_id)
                records[xc_id] = {
                    "status": "ok",
                    "fetched_at_utc": _utc_now(),
                    "recording": recording,
                    "error": "",
                }
            except XenoCantoFatalApiError:
                raise
            except XenoCantoRecordUnavailableError as exc:
                records[xc_id] = {
                    "status": "unavailable",
                    "fetched_at_utc": _utc_now(),
                    "recording": {},
                    "error": str(exc),
                }
            except XenoCantoApiError as exc:
                records[xc_id] = {
                    "status": "error",
                    "fetched_at_utc": _utc_now(),
                    "recording": {},
                    "error": str(exc),
                }
            cache["updated_at_utc"] = _utc_now()
            if index % checkpoint_every == 0 or index == len(pending):
                atomic_write_json(destination, cache)
                print(f"Fetched {index}/{len(pending)} pending metadata records")
            if index < len(pending):
                time.sleep(request_interval_seconds)
        require_unchanged(manifest_path, manifest_sha256)
        cache["source_manifest_sha256"] = manifest_sha256
        cache["source_recording_ids_sha256"] = recording_ids_sha256
        cache["updated_at_utc"] = _utc_now()
        destination = atomic_write_json(destination, cache)
    return destination, cache


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _normalize_session_text(value: Any) -> str:
    """Normalize contributor and locality text without discarding non-Latin scripts."""
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(
        "".join(character if character.isalnum() else " " for character in text).split()
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _normalize_url(value: Any) -> str:
    text = str(value or "").strip()
    return f"https:{text}" if text.startswith("//") else text


_CC_LICENCE_PATH_PATTERN = re.compile(
    r"^/licenses/(?:by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd)/"
    r"(?:1\.0|2\.(?:0|1|5)|3\.0|4\.0)/?$",
    flags=re.IGNORECASE,
)
_CC_PUBLIC_DOMAIN_PATH_PATTERN = re.compile(
    r"^/publicdomain/(?:zero|mark)/1\.0/?$",
    flags=re.IGNORECASE,
)


def _is_recognized_cc_licence_uri(value: Any) -> bool:
    """Accept canonical Creative Commons licence and public-domain URIs only."""
    uri = _normalize_url(value)
    if not uri:
        return False
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme.casefold() not in {"http", "https"}:
        return False
    if parsed.hostname is None or parsed.hostname.casefold() not in {
        "creativecommons.org",
        "www.creativecommons.org",
    }:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if parsed.query or parsed.fragment:
        return False
    return bool(
        _CC_LICENCE_PATH_PATTERN.fullmatch(parsed.path)
        or _CC_PUBLIC_DOMAIN_PATH_PATTERN.fullmatch(parsed.path)
    )


def _licence_validation_status(value: Any) -> str:
    if not _normalize_url(value):
        return "missing"
    return "recognized_cc" if _is_recognized_cc_licence_uri(value) else "unrecognized_uri"


def _binomial_identity(value: Any) -> tuple[str, str] | None:
    parts = _normalize_text(value).split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _same_binomial_identity(left: Any, right: Any) -> bool:
    left_identity = _binomial_identity(left)
    return left_identity is not None and left_identity == _binomial_identity(right)


def _scientific_name(recording: dict[str, Any]) -> str:
    return " ".join(
        part for part in (recording.get("gen"), recording.get("sp"), recording.get("ssp")) if part
    )


def _coordinates(row: dict[str, str]) -> tuple[float, float] | None:
    try:
        latitude = float(row.get("latitude") or "")
        longitude = float(row.get("longitude") or "")
    except ValueError:
        return None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    return latitude, longitude


def _valid_recorded_date(value: Any) -> bool:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return False
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _distance_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    delta_latitude = lat2 - lat1
    delta_longitude = lon2 - lon1
    value = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_longitude / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _flag_session_review(row: dict[str, str], reason: str) -> None:
    reasons = set(filter(None, row.get("session_review_reason", "").split(";")))
    reasons.add(reason)
    row["session_review_flag"] = "true"
    row["session_review_reason"] = ";".join(sorted(reasons))
    apply_qc_reason(row, f"{reason}_manual_review", "manual_review")


def assign_session_groups(rows: list[dict[str, str]], coordinate_radius_km: float = 1.0) -> None:
    """Build conservative session components before any recording-level split."""
    if coordinate_radius_km <= 0:
        raise ValueError("coordinate_radius_km must be positive")
    ordered = sorted(
        (row for row in rows if row.get("metadata_status") == "ok"),
        key=lambda row: row["recording_id"],
    )
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    buckets: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(ordered):
        row["session_group"] = ""
        row["session_review_flag"] = "false"
        row["session_review_reason"] = ""
        recordist = _normalize_session_text(row.get("recordist"))
        recorded_date = row.get("recorded_date", "").strip()
        date = recorded_date if _valid_recorded_date(recorded_date) else ""
        if not recordist:
            _flag_session_review(row, "session_recordist_missing")
        if not recorded_date:
            _flag_session_review(row, "session_date_missing")
        elif not date:
            _flag_session_review(row, "session_date_invalid")
        coordinates_present = bool(
            row.get("latitude", "").strip() or row.get("longitude", "").strip()
        )
        if coordinates_present and _coordinates(row) is None:
            _flag_session_review(row, "session_coordinates_invalid")
        buckets.setdefault((recordist or "unknown_recordist", date or "missing_date"), []).append(
            index
        )

    for bucket_indices in buckets.values():
        for offset, left in enumerate(bucket_indices):
            left_locality = _normalize_session_text(ordered[left].get("locality"))
            left_coordinates = _coordinates(ordered[left])
            for right in bucket_indices[offset + 1 :]:
                right_locality = _normalize_session_text(ordered[right].get("locality"))
                right_coordinates = _coordinates(ordered[right])
                same_locality = bool(left_locality and left_locality == right_locality)
                nearby = bool(
                    left_coordinates
                    and right_coordinates
                    and _distance_km(left_coordinates, right_coordinates) <= coordinate_radius_km
                )
                either_location_missing = not (left_locality or left_coordinates) or not (
                    right_locality or right_coordinates
                )
                if same_locality or nearby or either_location_missing:
                    union(left, right)

    id_to_index = {row["xc_id"]: index for index, row in enumerate(ordered)}
    for index, row in enumerate(ordered):
        remarks = row.get("remarks", "")
        if not SAME_INDIVIDUAL_PATTERN.search(remarks):
            continue
        references = sorted(set(XC_REFERENCE_PATTERN.findall(remarks)), key=int)
        in_corpus_references = [
            reference
            for reference in references
            if reference in id_to_index and id_to_index[reference] != index
        ]
        valid_references = [
            reference
            for reference in in_corpus_references
            if ordered[id_to_index[reference]].get("species_common_name")
            == row.get("species_common_name")
        ]
        cross_species_references = sorted(
            set(in_corpus_references) - set(valid_references), key=int
        )
        for reference in valid_references:
            union(index, id_to_index[reference])
        if cross_species_references:
            _flag_session_review(row, "session_same_individual_cross_species")
            apply_qc_reason(
                row,
                "metadata_cross_species_same_individual_manual_review",
                "manual_review",
            )
        if not in_corpus_references or len(in_corpus_references) != len(references):
            _flag_session_review(row, "session_same_individual_unresolved")

    components: dict[int, list[dict[str, str]]] = {}
    for index, row in enumerate(ordered):
        components.setdefault(find(index), []).append(row)
    for component_rows in components.values():
        component_ids = sorted(row["recording_id"] for row in component_rows)
        session = f"session:{sha256_json(component_ids)[:16]}"
        for row in component_rows:
            row["session_group"] = session


def _secondary_identities(secondary: str) -> set[str]:
    identities = {_normalize_text(secondary)}
    identities.update(
        _normalize_text(segment)
        for segment in re.split(r"[()\[\],;/|]", secondary)
        if _normalize_text(segment)
    )
    identities.discard("")
    return identities


def _labels_overlap(secondary: str, target_labels: set[str]) -> bool:
    return bool(_secondary_identities(secondary) & target_labels)


def _secondary_matches_different_target(
    secondary: str,
    target_labels: set[str],
    own_labels: set[str],
) -> bool:
    return bool(_secondary_identities(secondary) & (target_labels - own_labels))


def _configured_target_labels(config: dict[str, Any]) -> set[str]:
    """Return every study species label that must remain absent from backgrounds."""
    species = [
        *(config.get("known_species") or []),
        *(config.get("unknown_species") or []),
        *(config.get("fallback_unknown_species") or []),
    ]
    return {
        normalized
        for entry in species
        for value in (entry.get("common_name"), entry.get("scientific_name"))
        if (normalized := _normalize_text(value))
    }


def _apply_metadata(row: dict[str, str], entry: dict[str, Any], target_labels: set[str]) -> None:
    if entry.get("status") == "unavailable":
        row["metadata_status"] = "unavailable"
        row["metadata_error"] = str(entry.get("error") or "recording_unavailable")[:1000]
        row["identity_validation_status"] = "not_validated"
        row["licence_validation_status"] = "not_validated"
        apply_qc_reason(row, "metadata_record_unavailable", "exclude")
        return
    if entry.get("status") != "ok" or not isinstance(entry.get("recording"), dict):
        row["metadata_status"] = "error"
        row["metadata_error"] = str(entry.get("error") or "metadata_missing")[:1000]
        row["identity_validation_status"] = "not_validated"
        row["licence_validation_status"] = "not_validated"
        apply_qc_reason(row, "metadata_fetch_failed", "manual_review")
        return

    recording = entry["recording"]
    secondary = _string_list(recording.get("also"))
    api_scientific_name = _scientific_name(recording)
    api_group = str(recording.get("grp") or recording.get("group") or "")
    licence = _normalize_url(recording.get("lic"))
    licence_validation_status = _licence_validation_status(licence)
    vocalisation = _string_list(recording.get("type"))
    longitude = recording.get("lon")
    if longitude in (None, ""):
        longitude = recording.get("lng")
    identity_matches = _same_binomial_identity(
        row.get("scientific_name"),
        api_scientific_name,
    )
    group_matches = _normalize_text(api_group) == "birds"
    row.update(
        {
            "metadata_status": "ok",
            "metadata_fetched_at_utc": str(entry.get("fetched_at_utc") or ""),
            "metadata_query_version": API_VERSION,
            "metadata_error": "",
            "primary_label": str(recording.get("en") or ""),
            "api_scientific_name": api_scientific_name,
            "api_group": api_group,
            "secondary_labels": json.dumps(secondary, ensure_ascii=True, separators=(",", ":")),
            "target_secondary_labels": "[]",
            "recordist": str(recording.get("rec") or ""),
            "country": str(recording.get("cnt") or ""),
            "locality": str(recording.get("loc") or ""),
            "latitude": str(recording.get("lat") or ""),
            "longitude": str(longitude if longitude is not None else ""),
            "recorded_date": str(recording.get("date") or ""),
            "recorded_time": str(recording.get("time") or ""),
            "quality": str(recording.get("q") or ""),
            "vocalisation_type": ";".join(vocalisation),
            "remarks": str(recording.get("rmk") or ""),
            "recording_device": str(recording.get("dvc") or ""),
            "microphone": str(recording.get("mic") or ""),
            "recording_method": str(recording.get("method") or ""),
            "playback_used": str(recording.get("playback-used") or ""),
            "automatic_recording": str(recording.get("auto") or ""),
            "uploaded_date": str(recording.get("uploaded") or ""),
            "api_recording_url": _normalize_url(recording.get("url")),
            "api_sample_rate_hz": str(recording.get("smp") or ""),
            "licence": licence,
            "licence_status": "recorded" if licence else "missing",
            "licence_validation_status": licence_validation_status,
            "identity_validation_status": (
                "exact_match"
                if identity_matches and group_matches
                else "group_mismatch_or_missing"
                if identity_matches
                else "mismatch_or_missing"
            ),
            "source_audio_url": _normalize_url(recording.get("file")),
        }
    )
    row["attribution"] = (
        f"{row['recordist']}, {row['recording_id']}. "
        f"Accessible at www.xeno-canto.org/{row['xc_id']}."
    )

    if not identity_matches:
        apply_qc_reason(row, "metadata_primary_mismatch_manual_review", "manual_review")
    elif not group_matches:
        apply_qc_reason(row, "metadata_group_mismatch_manual_review", "manual_review")

    own_labels = {
        _normalize_text(row["species_common_name"]),
        _normalize_text(row["scientific_name"]),
    }
    target_secondary = [
        label
        for label in secondary
        if _secondary_matches_different_target(label, target_labels, own_labels)
    ]
    row["target_secondary_labels"] = json.dumps(
        target_secondary, ensure_ascii=True, separators=(",", ":")
    )
    if target_secondary:
        apply_qc_reason(row, "target_species_in_secondary_labels", "exclude")
    if licence_validation_status == "missing":
        apply_qc_reason(row, "licence_missing_manual_review", "manual_review")
    elif licence_validation_status != "recognized_cc":
        apply_qc_reason(row, "licence_unrecognized_manual_review", "manual_review")

    if row.get("local_qc_status") == "pending_metadata":
        row["local_qc_status"] = "include"


def enrich_manifest_from_cache(
    config_path: str | Path,
    local_manifest_path: str | Path,
    cache_path: str | Path,
    output_path: str | Path,
    licence_path: str | Path,
    summary_path: str | Path,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    config = load_toml(config_path)
    local_manifest = resolve_project_path(local_manifest_path)
    cache_file = resolve_project_path(cache_path)
    destination = require_safe_output(output_path)
    licence_destination = require_safe_output(licence_path)
    summary_destination = require_safe_output(summary_path)
    for path in (destination, licence_destination, summary_destination):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")

    target_labels = _configured_target_labels(config)
    with project_lock("metadata_enrichment"):
        rows, local_manifest_sha256 = read_csv_snapshot(local_manifest)
        cache, cache_sha256 = load_metadata_cache_snapshot(cache_file)
        records = cache["records"]
        xc_ids = sorted({row["xc_id"] for row in rows}, key=int)
        if cache.get("source_manifest_sha256") != local_manifest_sha256:
            raise ValueError("Metadata cache was not fetched from this exact local manifest")
        if cache.get("source_recording_ids_sha256") != sha256_json(xc_ids):
            raise ValueError("Metadata cache recording set does not match the local manifest")
        if set(records) != set(xc_ids):
            raise ValueError("Metadata cache must contain one entry for every local recording")
        for row in rows:
            _apply_metadata(row, records.get(row["xc_id"], {}), target_labels)
        assign_session_groups(
            rows,
            coordinate_radius_km=float(config["session_grouping"]["coordinate_radius_km"]),
        )

        rows.sort(key=lambda row: (row["species_folder"], int(row["xc_id"])))
        require_unchanged(local_manifest, local_manifest_sha256)
        require_unchanged(cache_file, cache_sha256)
        destination = atomic_write_csv(destination, rows, ENRICHED_MANIFEST_FIELDS)
        licence_rows = [{field: row.get(field, "") for field in LICENCE_FIELDS} for row in rows]
        licence_destination = atomic_write_csv(
            licence_destination,
            licence_rows,
            LICENCE_FIELDS,
        )
        qc_counts = Counter(row["local_qc_status"] for row in rows)
        metadata_counts = Counter(row["metadata_status"] for row in rows)
        identity_validation_counts = Counter(row["identity_validation_status"] for row in rows)
        licence_validation_counts = Counter(row["licence_validation_status"] for row in rows)
        summary = {
            "schema_version": "1.0",
            "source_local_manifest_sha256": local_manifest_sha256,
            "source_metadata_cache_sha256": cache_sha256,
            "enriched_manifest_sha256": sha256_file(destination),
            "recordings": len(rows),
            "metadata_statuses": dict(sorted(metadata_counts.items())),
            "identity_validation_statuses": dict(sorted(identity_validation_counts.items())),
            "licence_validation_statuses": dict(sorted(licence_validation_counts.items())),
            "local_qc_statuses": dict(sorted(qc_counts.items())),
            "target_secondary_exclusions": sum(
                "target_species_in_secondary_labels" in row["exclusion_reasons"] for row in rows
            ),
            "session_review_flags": sum(row["session_review_flag"] == "true" for row in rows),
            "licence_missing": sum(not row["licence"] for row in rows),
            "ready_for_manual_review": (
                metadata_counts.get("ok", 0) + metadata_counts.get("unavailable", 0) == len(rows)
                and metadata_counts.get("error", 0) == 0
            ),
            "ready_for_split": (
                metadata_counts.get("ok", 0) == len(rows)
                and qc_counts.get("pending_metadata", 0) == 0
                and qc_counts.get("manual_review", 0) == 0
            ),
        }
        atomic_write_json(summary_destination, summary)
    return destination, summary


def api_key_from_environment(variable_name: str = "XENO_CANTO_API_KEY") -> str:
    api_key = os.environ.get(variable_name, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {variable_name} in the active shell before fetching metadata")
    return api_key
