import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.wanikani.com/v2"
WANIKANI_REVISION = "20170710"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Endpoints included in the raw pull pipeline.
ENDPOINTS = [
    "review_statistics",
    "subjects",
    "assignments",
    "summary",
    "user",
    "level_progressions",
    "resets",
    "spaced_repetition_systems",
]

# Only collection endpoints accept the updated_after query parameter.
UPDATED_AFTER_CAPABLE_ENDPOINTS = {
    "review_statistics",
    "subjects",
    "assignments",
    "level_progressions",
    "resets",
    "spaced_repetition_systems",
}


def load_json(path: Path, default):
    """Load JSON from disk and return a default value when missing."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    """Write JSON data to disk, ensuring parent folders exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_iso8601(value: str | None) -> str | None:
    """Validate and normalize an ISO 8601 timestamp string for API filters."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_headers(api_token: str) -> dict[str, str]:
    """Construct request headers with auth and explicit API revision."""
    return {
        "Authorization": f"Bearer {api_token}",
        "Wanikani-Revision": WANIKANI_REVISION,
    }


def calculate_retry_wait_seconds(response: requests.Response, attempt: int) -> float:
    """Calculate backoff delay using rate-limit headers when available."""
    if response.status_code == 429:
        reset_header = response.headers.get("RateLimit-Reset")
        if reset_header and reset_header.isdigit():
            reset_ts = int(reset_header)
            wait_seconds = max(0, reset_ts - int(time.time())) + 1
            return float(wait_seconds)

    # Exponential backoff capped at 30s.
    return float(min(30, 2 ** max(0, attempt - 1)))


def maybe_throttle(response: requests.Response) -> None:
    """Respect near-limit responses by pausing until the reset boundary."""
    remaining_header = response.headers.get("RateLimit-Remaining")
    reset_header = response.headers.get("RateLimit-Reset")

    if remaining_header and remaining_header.isdigit() and int(remaining_header) <= 1:
        if reset_header and reset_header.isdigit():
            wait_seconds = max(0, int(reset_header) - int(time.time())) + 1
            if wait_seconds > 0:
                print(f"Approaching rate limit. Sleeping {wait_seconds}s.")
                time.sleep(wait_seconds)
                return

    # Small pause between requests helps smooth bursty pagination.
    time.sleep(0.1)


def request_with_retry(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    params: dict[str, str] | None,
    timeout_seconds: int,
    max_retries: int,
) -> requests.Response:
    """Send a GET request with retry behavior for transient failures."""
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, headers=headers, params=params, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_exception = exc
            wait_seconds = min(30, 2 ** max(0, attempt - 1))
            print(f"Request error on attempt {attempt}/{max_retries}: {exc}. Retrying in {wait_seconds}s.")
            time.sleep(wait_seconds)
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            wait_seconds = calculate_retry_wait_seconds(response, attempt)
            print(
                f"Retryable status {response.status_code} on attempt {attempt}/{max_retries}. "
                f"Retrying in {wait_seconds:.1f}s."
            )
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        return response

    if last_exception:
        raise RuntimeError(f"Request failed after {max_retries} retries: {last_exception}")
    raise RuntimeError(f"Request failed after {max_retries} retries with retryable status codes.")


def fetch_endpoint_data(
    session: requests.Session,
    endpoint: str,
    headers: dict[str, str],
    updated_after: str | None,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[object, dict]:
    """Fetch all pages for an endpoint and return payload with pull metadata."""
    url = f"{BASE_URL}/{endpoint}"
    params = {"updated_after": updated_after} if updated_after else None

    all_data = []
    single_object_data = None
    saw_list_data = False
    latest_data_updated_at = None

    while url:
        print(f"Fetching: {url}{' with updated_after=' + updated_after if params else ''}")
        response = request_with_retry(
            session=session,
            url=url,
            headers=headers,
            params=params,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        payload = response.json()

        if payload.get("data_updated_at"):
            latest_data_updated_at = payload.get("data_updated_at")

        page_data = payload.get("data")
        if isinstance(page_data, list):
            saw_list_data = True
            all_data.extend(page_data)
        elif page_data is not None:
            single_object_data = page_data

        url = (payload.get("pages") or {}).get("next_url")
        params = None
        if url:
            maybe_throttle(response)

    result = all_data if saw_list_data else single_object_data
    metadata = {
        "endpoint": endpoint,
        "is_collection": saw_list_data,
        "record_count": len(all_data) if saw_list_data else (1 if single_object_data is not None else 0),
        "data_updated_at": latest_data_updated_at,
        "pulled_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "updated_after_used": updated_after,
    }
    return result, metadata


def upsert_collection_records(existing_data: list, incoming_data: list) -> list:
    """Merge incremental collection updates into an existing full collection by id."""
    by_id: dict[int, dict] = {}
    without_id: list[dict] = []

    for row in existing_data:
        if isinstance(row, dict) and isinstance(row.get("id"), int):
            by_id[row["id"]] = row
        else:
            without_id.append(row)

    for row in incoming_data:
        if isinstance(row, dict) and isinstance(row.get("id"), int):
            by_id[row["id"]] = row
        else:
            without_id.append(row)

    merged = [by_id[key] for key in sorted(by_id.keys())]
    merged.extend(without_id)
    return merged


def determine_updated_after(
    endpoint: str,
    state: dict,
    force_updated_after: str | None,
    full_refresh: bool,
) -> str | None:
    """Determine updated_after for a collection endpoint using pull state."""
    if full_refresh:
        return None

    if endpoint not in UPDATED_AFTER_CAPABLE_ENDPOINTS:
        return None

    if force_updated_after:
        return force_updated_after

    endpoint_state = (state.get("endpoints") or {}).get(endpoint, {})
    return endpoint_state.get("last_data_updated_at")


def main() -> None:
    """Pull raw WaniKani endpoints with incremental updates and retries."""
    parser = argparse.ArgumentParser(description="Pull and persist raw WaniKani API data.")
    parser.add_argument("--output-dir", default="data/raw", help="Directory where latest raw files are written.")
    parser.add_argument("--full-refresh", action="store_true", help="Disable updated_after and pull full collections.")
    parser.add_argument(
        "--updated-after",
        default=None,
        help="Force a specific updated_after timestamp for collection endpoints (ISO 8601).",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retry attempts per request.")
    args = parser.parse_args()

    load_dotenv()
    api_token = os.getenv("WANIKANI_API_TOKEN")
    if not api_token:
        raise RuntimeError("WANIKANI_API_TOKEN is not set. Add it to your .env file.")

    forced_updated_after = parse_iso8601(args.updated_after)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "pull_state.json"
    state = load_json(state_path, default={"endpoints": {}})

    headers = build_headers(api_token)

    with requests.Session() as session:
        for endpoint in ENDPOINTS:
            updated_after = determine_updated_after(
                endpoint=endpoint,
                state=state,
                force_updated_after=forced_updated_after,
                full_refresh=args.full_refresh,
            )

            pulled_data, metadata = fetch_endpoint_data(
                session=session,
                endpoint=endpoint,
                headers=headers,
                updated_after=updated_after,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )

            output_file = output_dir / f"{endpoint}.json"

            final_data = pulled_data
            if isinstance(pulled_data, list) and updated_after and output_file.exists():
                existing_data = load_json(output_file, default=[])
                if isinstance(existing_data, list):
                    final_data = upsert_collection_records(existing_data, pulled_data)
                    metadata["record_count_after_upsert"] = len(final_data)

            save_json(output_file, final_data)

            endpoint_state = {
                "last_successful_pull_at": metadata.get("pulled_at"),
                "last_data_updated_at": metadata.get("data_updated_at"),
                "last_record_count": metadata.get("record_count_after_upsert", metadata.get("record_count", 0)),
                "last_updated_after_used": metadata.get("updated_after_used"),
            }
            state.setdefault("endpoints", {})[endpoint] = endpoint_state

            print(
                f"Saved {endpoint} to {output_file} "
                f"({endpoint_state['last_record_count']} records, updated_after={updated_after})"
            )

    save_json(state_path, state)
    print(f"Saved pull state to {state_path}")


if __name__ == "__main__":
    main()