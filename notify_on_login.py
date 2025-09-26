import asyncio
import time
import argparse
import json
import os
import random
import requests
import csv
from http.cookiejar import Cookie
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
from vrchatapi import ApiClient
from vrchatapi.configuration import Configuration
from vrchatapi import AuthenticationApi, FriendsApi
from vrchatapi.exceptions import UnauthorizedException, ApiException
from vrchatapi.models.two_factor_auth_code import TwoFactorAuthCode
from vrchatapi.models.two_factor_email_code import TwoFactorEmailCode
from datetime import datetime, timezone
from pathlib import Path
import aiohttp
from urllib.parse import urlparse

load_dotenv(".env")

USERNAME = os.getenv("VRCHAT_USERNAME")
PASSWORD = os.getenv("VRCHAT_PASSWORD")
EMAIL = os.getenv("VRCHAT_EMAIL")
contact_info = EMAIL if EMAIL else "no-contact@example.com"
USER_AGENT = f"VRCNotify/1.0 ({contact_info})"

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

VRC_FRIEND_NAMES = os.getenv("VRC_FRIEND_NAMES", "")
FRIEND_NAMES = [name.strip() for name in VRC_FRIEND_NAMES.split(",") if name.strip()]

CHECK_INTERVAL = 60  # seconds
PAUSE_ON_CRITICAL_ERRROR = 3600
PAUSE_ON_AUTH_ERROR = 271
RETRY_AFTER_BUFFER = 5  # seconds padding beyond Retry-After header
NOTE_REFRESH_SECONDS = 3600  # one hour cooldown for friend note lookups

STATE_FILE = Path("friend_activity.json")
CSV_LOG_FILE = Path("user_activity_log.csv")
COOKIES_FILE = Path("auth_cookies.json")

# ---------- NEW: favorites persistence ----------
FAV_AVATARS_FILE = Path("favorite_avatars.json")
FAV_AVATAR_FEATURES = [
    "notes",
    "rank",  # score 1-10
    "gender",
    "VRCFT",
    "quest",
    "natural",
    "fun",
    "fly",
    "world_drop",
    "seat",
    "want",
    "add_date",
    "seek"
]

# NEW: avatar image cache directory
AVATAR_IMG_DIR = Path("avatar_images")
AVATAR_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Optional typed APIs (guarded)
try:
    from vrchatapi import AvatarsApi, FavoritesApi, WorldsApi, UsersApi
except Exception:
    AvatarsApi = None
    FavoritesApi = None
    WorldsApi = None
    UsersApi = None

# cache resolved world and avatar names to limit API calls
WORLD_NAME_CACHE = {}
AVATAR_NAME_CACHE = {}

LAST_NOTIFICATION_DATE = None


# Map platform identifiers to their human-friendly names.
def normalize_platform_name(platform_value: str) -> str:
    if not platform_value:
        return ""
    value = platform_value.strip()
    if not value:
        return ""
    mappings = {
        "android": "Quest",
    }
    return mappings.get(value.lower(), value)


def format_notification_timestamp(current_dt: datetime) -> str:
    global LAST_NOTIFICATION_DATE
    date_part = current_dt.strftime("%Y-%m-%d")
    time_part = current_dt.strftime("%H:%M:%S")
    if LAST_NOTIFICATION_DATE != date_part:
        LAST_NOTIFICATION_DATE = date_part
        return f"{date_part} {time_part}"
    return time_part


def _presence_field(presence, key):
    if presence is None:
        return None
    if isinstance(presence, dict):
        return presence.get(key)
    return getattr(presence, key, None)


def get_presence_location(user) -> str:
    if not user:
        return None
    presence = getattr(user, "presence", None)
    world = _presence_field(presence, "world")
    instance = _presence_field(presence, "instance")
    if world and instance:
        return f"{world}:{instance}"
    if world:
        return world
    if instance:
        return instance
    fallback = getattr(user, "location", None)
    return fallback


def determine_location_type(location: str) -> str:
    if not location:
        return "unknown"
    location_lower = location.lower()
    if location_lower == "offline":
        return "offline"
    if location_lower == "private" or "~private" in location_lower:
        return "private"
    if "~friends" in location_lower:
        return "friends"
    if "~hidden" in location_lower:
        return "friends+"
    if "~group" in location_lower:
        return "group"
    if location_lower.startswith("wrld_") and "~" not in location_lower:
        return "public"
    if "~public" in location_lower:
        return "public"
    return "instance"


def _format_duration_seconds(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ensure_activity_record(activity: dict, name: str, notify_flag: bool):
    record = activity.get(name)
    if not isinstance(record, dict):
        record = {}

    record.setdefault("online", False)
    record["notify"] = bool(notify_flag)
    record.setdefault("last_update", None)
    record.setdefault("last_platform", "")
    record.setdefault("current_location", "offline")
    record.setdefault("last_location", "")
    record.setdefault("status_description", "")
    record.setdefault("last_avatar_name", None)
    record.setdefault("online_since", None)
    record.setdefault("friend_id", None)
    record.setdefault("friend_note", None)

    history = record.get("meet_history")
    if not isinstance(history, list):
        history = []
    else:
        sanitized_history = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            entry_copy = dict(entry)
            other = entry_copy.get("other_friends", [])
            if isinstance(other, (set, tuple)):
                other = list(other)
            if not isinstance(other, list):
                other = []
            sanitized_history.append({
                "datetime": entry_copy.get("datetime"),
                "location": entry_copy.get("location"),
                "location_type": entry_copy.get("location_type", "unknown"),
                "duration": str(entry_copy.get("duration", "00:00:00")),
                "other_friends": sorted({str(o) for o in other if o}),
            })
        history = sanitized_history
    record["meet_history"] = history

    active_meet = record.get("active_meet")
    if isinstance(active_meet, dict):
        active_copy = dict(active_meet)
        other = active_copy.get("other_friends", [])
        if isinstance(other, (set, tuple)):
            other = list(other)
        if not isinstance(other, list):
            other = []
        active_copy["other_friends"] = sorted({str(o) for o in other if o})
        try:
            active_copy["start_ts"] = float(active_copy.get("start_ts", 0))
        except (TypeError, ValueError):
            active_copy["start_ts"] = 0.0
        if "history_index" in active_copy:
            try:
                active_copy["history_index"] = int(active_copy["history_index"])
            except (TypeError, ValueError):
                active_copy["history_index"] = None
        record["active_meet"] = active_copy
    else:
        record["active_meet"] = None

    activity[name] = record
    return record


def start_meet_session(record: dict, start_dt: datetime, raw_location: str,
                       location_label: str, location_type: str, other_friends):
    start_ts = start_dt.timestamp()
    other_sorted = sorted({friend for friend in other_friends if friend})
    entry = {
        "datetime": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "location": location_label,
        "location_type": location_type,
        "duration": "00:00:00",
        "other_friends": other_sorted,
    }
    history = record.setdefault("meet_history", [])
    history.append(entry)
    record["active_meet"] = {
        "start_ts": start_ts,
        "location": raw_location,
        "location_label": location_label,
        "location_type": location_type,
        "other_friends": other_sorted,
        "history_index": len(history) - 1,
    }


def update_meet_session(record: dict, now_dt: datetime, other_friends):
    active = record.get("active_meet")
    if not active:
        return
    other_set = set(active.get("other_friends", []))
    for friend in other_friends:
        if friend:
            other_set.add(friend)
    other_sorted = sorted(other_set)
    active["other_friends"] = other_sorted

    history_index = active.get("history_index")
    history = record.get("meet_history", [])
    if history_index is not None and 0 <= history_index < len(history):
        entry = history[history_index]
        entry["other_friends"] = other_sorted
        start_ts = active.get("start_ts", now_dt.timestamp())
        duration_str = _format_duration_seconds(now_dt.timestamp() - start_ts)
        entry["duration"] = duration_str


def end_meet_session(record: dict, now_dt: datetime):
    active = record.get("active_meet")
    if not active:
        return None
    history_index = active.get("history_index")
    history = record.get("meet_history", [])
    entry = None
    if history_index is not None and 0 <= history_index < len(history):
        entry = history[history_index]
    record["active_meet"] = None
    if entry is None:
        return None
    other = entry.get("other_friends") or []
    if isinstance(other, (set, tuple)):
        other = list(other)
    info = {
        "datetime": entry.get("datetime"),
        "location": entry.get("location"),
        "location_type": entry.get("location_type", "unknown"),
        "duration": entry.get("duration", "00:00:00"),
        "other_friends": [str(o) for o in other if o],
        "location_label": active.get("location_label") or entry.get("location"),
        "raw_location": active.get("location"),
    }
    return info


# ---------- Friend-activity persistence (unchanged) ----------
def load_activity():
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
    else:
        data = {}

    activity = {}
    for name, record in data.items():
        if not isinstance(name, str):
            continue
        activity[name] = record

    for name in list(activity.keys()):
        ensure_activity_record(activity, name, name in FRIEND_NAMES)

    for name in FRIEND_NAMES:
        ensure_activity_record(activity, name, True)

    return activity


def save_activity(activity):
    serializable = {}
    for name, record in activity.items():
        if not isinstance(name, str):
            continue
        if not isinstance(record, dict):
            continue
        rec_copy = dict(record)
        active_meet = rec_copy.pop("active_meet", None)
        history = rec_copy.get("meet_history", [])
        sanitized_history = []
        if isinstance(history, list):
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                entry_copy = dict(entry)
                other = entry_copy.get("other_friends", [])
                if isinstance(other, (set, tuple)):
                    other = list(other)
                if not isinstance(other, list):
                    other = []
                entry_copy["other_friends"] = sorted({str(o) for o in other if o})
                entry_copy["duration"] = str(entry_copy.get("duration", "00:00:00"))
                entry_copy.setdefault("location_type", "unknown")
                sanitized_history.append(entry_copy)
        rec_copy["meet_history"] = sanitized_history
        if active_meet:
            # keep minimal information for debugging reference only if needed
            active_copy = dict(active_meet)
            other = active_copy.get("other_friends", [])
            if isinstance(other, (set, tuple)):
                other = list(other)
            if not isinstance(other, list):
                other = []
            active_copy["other_friends"] = sorted({str(o) for o in other if o})
            rec_copy["active_meet"] = active_copy
        serializable[name] = rec_copy

    with STATE_FILE.open("w") as f:
        json.dump(serializable, f, indent=4)


def log_activity(user_name: str, event: str, *, location: str = "", platform: str = "",
                 avatar: str = "", status: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = CSV_LOG_FILE.exists()
    with CSV_LOG_FILE.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or csvfile.tell() == 0:
            writer.writerow(["timestamp", "user", "event", "location", "platform", "avatar", "status"])
        writer.writerow([
            timestamp,
            user_name,
            event,
            location or "",
            platform or "",
            avatar or "",
            status or "",
        ])


# single session with required UA for VRChat CDN
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": USER_AGENT})


async def send_discord_message(message: str, image_url: str = None, image_urls=None, image_entries=None, print_to_console=False):
    if print_to_console:
        print(message)
    if not WEBHOOK_URL:
        return
    # Normalize entries preference: entries override explicit URLs
    entries = []
    if image_entries:
        for entry in image_entries:
            if not entry:
                continue
            url = entry.get("url") if isinstance(entry, dict) else None
            name = entry.get("name") if isinstance(entry, dict) else None
            if url:
                entries.append({"name": name, "url": url})
    else:
        urls = []
        if image_urls:
            urls.extend([u for u in image_urls if u])
        if image_url:
            urls.append(image_url)
        if urls:
            unique_urls = []
            seen = set()
            for url in urls:
                if url in seen:
                    continue
                seen.add(url)
                unique_urls.append(url)
            entries = [{"name": None, "url": url} for url in unique_urls]

    embeds = []
    if entries:
        unique_entries = []
        seen_urls = set()
        for entry in entries:
            url = entry["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            unique_entries.append(entry)
        entries = unique_entries

        max_embeds = 10
        if len(entries) > max_embeds:
            entries = random.sample(entries, max_embeds)

        for entry in entries:
            embed = {"thumbnail": {"url": entry["url"]}}
            if entry.get("name"):
                embed["title"] = entry["name"]
            embeds.append(embed)

    data = {"content": message}
    if embeds:
        data["embeds"] = embeds

    sent = False
    while not sent:
        try:
            resp = requests.post(WEBHOOK_URL, json=data, timeout=15)
            sent = True
        except Exception as ex:
            print(f"Exception sending discord: {message} {ex}")
            await asyncio.sleep(2)


async def send_telegram_message(message: str, image_url: str = None):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)


async def send_notification(message: str, image_url: str = None, image_urls=None, image_entries=None):
    await send_discord_message(message, image_url=image_url, image_urls=image_urls, image_entries=image_entries)
    await send_telegram_message(message, image_url)


async def send_meet_end_notification(friend_name: str, meet_info: dict):
    if not meet_info:
        return
    lines = [f"Meet ended with {friend_name}."]
    start_time = meet_info.get("datetime")
    if start_time:
        lines.append(f"Started: {start_time}")
    location_label = meet_info.get("location_label") or meet_info.get("location") or "unknown"
    location_type = meet_info.get("location_type") or "unknown"
    lines.append(f"Location: {location_label} ({location_type})")
    duration = meet_info.get("duration") or "00:00:00"
    lines.append(f"Duration: {duration}")
    others = [o for o in (meet_info.get("other_friends") or []) if o and o != friend_name]
    if others:
        lines.append("Also present: " + ", ".join(sorted(set(others))))
    # message = "\n".join(lines)
    message = f"{start_time} Met {friend_name} at {location_label} ({location_type}), duration: {duration}"
    if others:
        message += "\nWith " + ", ".join(sorted(set(others)))
    print(message)
    await send_notification(message=message)


def make_cookie(name, value):
    return Cookie(0, name, value, None, False, "api.vrchat.cloud", True, False, "/", False, False, 173106866300, False, None, None, {})


def load_cookies(api_client: ApiClient):
    if COOKIES_FILE.exists():
        try:
            with COOKIES_FILE.open("r") as f:
                cookies = json.load(f)
                if cookies.get("auth"):
                    api_client.rest_client.cookie_jar.set_cookie(make_cookie("auth", cookies.get("auth")))
                if cookies.get("twoFactorAuth"):
                    api_client.rest_client.cookie_jar.set_cookie(make_cookie("twoFactorAuth", cookies.get("twoFactorAuth")))
            print("Loaded auth cookie from disk.")
        except Exception as e:
            print(f"Error loading cookies: {e}")


def save_cookies(api_client: ApiClient):
    try:
        cookie_jar = api_client.rest_client.cookie_jar._cookies["api.vrchat.cloud"]["/"]
        with COOKIES_FILE.open("w") as f:
            json.dump(
                {
                    "auth": cookie_jar.get("auth").value if "auth" in cookie_jar else None,
                    "twoFactorAuth": cookie_jar.get("twoFactorAuth").value if "twoFactorAuth" in cookie_jar else None,
                },
                f,
            )
    except Exception as e:
        print(f"save_cookies: {e}")


def clear_saved_cookies():
    try:
        if COOKIES_FILE.exists():
            COOKIES_FILE.unlink()
            print("Cleared saved auth cookies.")
    except Exception as ex:
        print(f"Failed to remove cached auth cookies: {ex}")


def _extract_retry_after_header(exc):
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        return headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        try:
            return headers["Retry-After"]
        except Exception:
            return None


def compute_retry_after_seconds(exc, default_wait: int) -> int:
    header_value = _extract_retry_after_header(exc)
    if not header_value:
        return default_wait
    try:
        wait_seconds = float(header_value)
        if wait_seconds < 0:
            wait_seconds = 0
        return int(wait_seconds + RETRY_AFTER_BUFFER)
    except (TypeError, ValueError):
        try:
            target_dt = parsedate_to_datetime(str(header_value))
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=timezone.utc)
            wait_seconds = (target_dt - datetime.now(timezone.utc)).total_seconds()
            if wait_seconds < 0:
                wait_seconds = 0
            return int(wait_seconds + RETRY_AFTER_BUFFER)
        except Exception:
            return default_wait


async def sleep_with_retry_after(exc, default_wait: int, label: str = ""):
    wait_seconds = compute_retry_after_seconds(exc, default_wait)
    if label:
        print(f"{label} Waiting {wait_seconds} seconds before retrying.")
    await asyncio.sleep(wait_seconds)


class FriendMetadataManager:
    def __init__(self, users_api, activity_status: dict):
        self.users_api = users_api
        self.activity_status = activity_status
        self.cache = {}

    def get_metadata(self, friend_id: str, allow_fetch: bool = True):
        if not friend_id:
            return None
        if friend_id in self.cache:
            return self.cache[friend_id]
        if not allow_fetch or self.users_api is None:
            return None
        try:
            detail = self.users_api.get_user(friend_id)
        except Exception as ex:
            print(f"Failed to fetch friend metadata for {friend_id}: {ex}")
            self.cache[friend_id] = None
            return None
        display_name = getattr(detail, "display_name", None) or getattr(detail, "username", None)
        note_value = getattr(detail, "note", None)
        meta = {
            "display_name": display_name,
            "note": note_value if note_value is not None else "",
            "note_checked_at": time.time(),
        }
        self.cache[friend_id] = meta
        return meta

    def sync_from_current_user(self, current_user) -> bool:
        updated = False
        if current_user is None:
            return updated
        try:
            current_friend_ids = {fid for fid in (getattr(current_user, "friends", []) or []) if fid}
        except Exception:
            current_friend_ids = set()
        if not current_friend_ids:
            return updated

        known_friend_ids = {rec.get("friend_id") for rec in self.activity_status.values() if rec.get("friend_id")}
        missing_ids = current_friend_ids - known_friend_ids
        for friend_id in missing_ids:
            meta = self.get_metadata(friend_id, allow_fetch=True)
            if not meta or not meta.get("display_name"):
                continue
            name = meta["display_name"]
            record = ensure_activity_record(self.activity_status, name, name in FRIEND_NAMES)
            if record.get("friend_id") != friend_id:
                record["friend_id"] = friend_id
                updated = True
            note_value = meta.get("note", "")
            if record.get("friend_note") != note_value:
                record["friend_note"] = note_value
                updated = True

        for name, record in list(self.activity_status.items()):
            friend_id = record.get("friend_id")
            if not friend_id or friend_id not in current_friend_ids:
                continue

            if not record.get("online"):
                continue

            meta = self.cache.get(friend_id)
            refreshed = False
            if self._note_stale(meta):
                meta = self.get_metadata(friend_id, allow_fetch=True)
                refreshed = True

            if meta:
                display_name = meta.get("display_name")
                note_value = meta.get("note", "")
                if display_name and display_name != name:
                    other = self.activity_status.get(display_name)
                    if other is None:
                        other = ensure_activity_record(self.activity_status, display_name, display_name in FRIEND_NAMES)
                    if other.get("friend_id") != friend_id:
                        other["friend_id"] = friend_id
                        updated = True
                    if other.get("friend_note") != note_value:
                        other["friend_note"] = note_value
                        updated = True
                if record.get("friend_note") != note_value:
                    record["friend_note"] = note_value
                    updated = True
                if refreshed:
                    meta["note_checked_at"] = time.time()
            else:
                if record.get("friend_note") is None:
                    record["friend_note"] = ""
                    updated = True

        return updated

    def update_record_from_detail(self, name: str, friend_detail, record: dict) -> bool:
        if friend_detail is None:
            return False
        changed = False
        friend_id_value = getattr(friend_detail, "id", None)
        friend_note_hint = getattr(friend_detail, "note", None)
        cache_entry = None
        now_ts = time.time()

        if friend_id_value:
            cache_entry = self.cache.get(friend_id_value) or {}
            if cache_entry.get("display_name") != name:
                cache_entry["display_name"] = name
            if record.get("friend_id") != friend_id_value:
                record["friend_id"] = friend_id_value
                changed = True
        if friend_note_hint is None and friend_id_value:
            if cache_entry and cache_entry.get("note") is not None:
                friend_note_hint = cache_entry.get("note")
            else:
                meta = self.get_metadata(friend_id_value, allow_fetch=False)
                if meta and meta.get("note") is not None:
                    friend_note_hint = meta.get("note")
        if friend_note_hint is None and friend_id_value and record.get("friend_note") is None:
            meta = self.get_metadata(friend_id_value, allow_fetch=True)
            if meta:
                friend_note_hint = meta.get("note", "")
        if friend_note_hint is not None and record.get("friend_note") != friend_note_hint:
            record["friend_note"] = friend_note_hint
            changed = True
        if friend_id_value:
            cache_entry = self.cache.get(friend_id_value) or {}
            cache_entry.setdefault("display_name", name)
            if friend_note_hint is not None:
                cache_entry["note"] = friend_note_hint
            elif record.get("friend_note") is not None:
                cache_entry["note"] = record.get("friend_note")
            else:
                cache_entry.setdefault("note", "")
            cache_entry["note_checked_at"] = now_ts
            self.cache[friend_id_value] = cache_entry
        return changed

    @staticmethod
    def _note_stale(meta) -> bool:
        if not meta:
            return True
        checked_at = meta.get("note_checked_at")
        if not checked_at:
            return True
        return (time.time() - checked_at) >= NOTE_REFRESH_SECONDS


def next_hour_epoch():
    now = datetime.now()
    return time.time() + ((60 - now.minute - 1) * 60 + (60 - now.second))


async def handle_initial_online_snapshot(initial_online, activity_status, metadata_manager: FriendMetadataManager,
                                         avatars_api_instance, worlds_api_instance, api_client):
    if not initial_online:
        return

    initial_online_names = []
    initial_image_entries = []
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for friend in initial_online:
        name = friend.display_name
        raw_location = friend.location or "private"
        platform_raw = friend.last_platform or "unknown"
        platform_display = normalize_platform_name(platform_raw) or "unknown"
        formatted_location = format_location(raw_location, worlds_api_instance, api_client)
        avatar_name = get_friend_avatar_name(friend, avatars_api_instance, api_client)
        avatar_suffix = f" • {avatar_name}" if avatar_name else ""
        initial_online_names.append(f"{name}({platform_display}/{formatted_location}{avatar_suffix})")
        image_url = pick_friend_image_url(friend)
        if image_url:
            initial_image_entries.append({"name": name, "url": image_url})

        record = ensure_activity_record(activity_status, name, name in FRIEND_NAMES)
        record.update({
            "online": True,
            "last_update": now_ts,
            "last_platform": platform_display,
            "status_description": getattr(friend, "status_description", "") or "",
            "current_location": raw_location or "offline",
        })
        if avatar_name:
            record["last_avatar_name"] = avatar_name
        if raw_location and raw_location.lower() not in {"private", "offline"}:
            record["last_location"] = formatted_location
        record["online_since"] = now_ts
        record["active_meet"] = None

        metadata_manager.update_record_from_detail(name, friend, record)

        log_activity(
            name,
            "online",
            location=formatted_location,
            platform=platform_display,
            avatar=avatar_name or "",
            status=getattr(friend, "status_description", "") or "",
        )

    msg = "Initial online friends: " + ", ".join(initial_online_names)
    print(msg)
    await send_notification(msg, image_entries=initial_image_entries)
    save_activity(activity_status)


async def end_active_meets_for_offline_user(activity_status, loop_now_dt, now):
    for friend_name, friend_record in activity_status.items():
        if friend_record.get("active_meet"):
            ended_meet = end_meet_session(friend_record, loop_now_dt)
            if ended_meet:
                await send_meet_end_notification(friend_name, ended_meet)
            friend_record["last_update"] = now


def init_optional_apis(api_client):
    avatars_api_instance = None
    if 'AvatarsApi' in globals() and AvatarsApi is not None:
        try:
            avatars_api_instance = AvatarsApi(api_client)
        except Exception as ex:
            print(f"AvatarsApi init failed: {ex}")
            avatars_api_instance = None

    worlds_api_instance = None
    if 'WorldsApi' in globals() and WorldsApi is not None:
        try:
            worlds_api_instance = WorldsApi(api_client)
        except Exception as ex:
            print(f"WorldsApi init failed: {ex}")
            worlds_api_instance = None

    users_api_instance = None
    if 'UsersApi' in globals() and UsersApi is not None:
        try:
            users_api_instance = UsersApi(api_client)
        except Exception as ex:
            print(f"UsersApi init failed: {ex}")
            users_api_instance = None

    return avatars_api_instance, worlds_api_instance, users_api_instance


def log_periodic_online_set(now: str, online_set):
    if now[-4] == "0":
        print(f"{now} {','.join(online_set)}")


async def send_online_digest(now: str, online_friends, online_set, avatars_api_instance,
                             worlds_api_instance, api_client):
    online_entries = []
    others_online = []
    digest_image_entries = []
    for friend in online_friends:
        if friend.display_name not in FRIEND_NAMES:
            others_online.append(friend.display_name)
            continue
        name = friend.display_name
        location = friend.location or ""
        world_name = resolve_world_name_only(location, worlds_api_instance, api_client)
        avatar_name = get_friend_avatar_name(friend, avatars_api_instance, api_client)
        image_url = pick_friend_image_url(friend)
        if image_url:
            digest_image_entries.append({"name": name, "url": image_url})
        if world_name:
            if avatar_name:
                online_entries.append(f"{name} ({world_name} • {avatar_name})")
            else:
                online_entries.append(f"{name} ({world_name})")
        elif avatar_name:
            online_entries.append(f"{name} ({avatar_name})")
        else:
            online_entries.append(name)

    if not online_entries:
        online_entries = sorted(online_set)
    else:
        online_entries.sort()

    message = f"{now} Online:  {', '.join(online_entries)}"
    if others_online:
        message += f"\nOthers:  {', '.join(others_online)}"

    await send_notification(message=message, image_entries=digest_image_entries)


async def handle_favorites_rescan_if_due(api_client, fav_db, args, now: str, next_fav_scan):
    if time.time() < next_fav_scan:
        return fav_db, next_fav_scan

    try:
        fav_db_current = load_fav_db()
        current_favs = await fetch_current_favorite_avatars(
            api_client, fav_db_current, force_read=args.force_read_avatars
        )

        before = {aid for aid, rec in fav_db_current["avatars"].items() if rec.get("active")}
        fav_db = merge_favorites_snapshot(fav_db_current, current_favs)
        after = {aid for aid, rec in fav_db["avatars"].items() if rec.get("active")}
        added = sorted(after - before)
        removed = sorted(before - after)
        save_fav_db(fav_db)

        if added or removed:
            lines = []
            if added:
                names = [f"{fav_db['avatars'][aid]['name']}({aid})" for aid in added]
                lines.append("🟢 Favorites added: " + ", ".join(names))
            if removed:
                names = [f"{fav_db['avatars'][aid]['name']}({aid})" for aid in removed]
                lines.append("🔴 Favorites removed: " + ", ".join(names))
            print("\n".join(lines))
            await send_notification("Favorites changed:\n" + "\n".join(lines))
        else:
            print(f"{now} Favorites scan: no changes.")
    except Exception as ex:
        print(f"Hourly favorites scan failed: {ex}")

    return fav_db, next_hour_epoch()


async def monitor_loop(args, api_client, auth_api, friends_api_instance, avatars_api_instance,
                       worlds_api_instance, metadata_manager: FriendMetadataManager,
                       activity_status: dict, fav_db):
    next_fav_scan = next_hour_epoch()

    while True:
        loop_now_dt = datetime.now()
        now = loop_now_dt.strftime("%Y-%m-%d %H:%M:%S")

        try:
            online_friends = friends_api_instance.get_friends()
        except Exception as ex:
            print(f"Critical error: {ex}")
            await send_notification(message=f"Critical error: {ex}, waiting 1 hour.")
            await asyncio.sleep(PAUSE_ON_CRITICAL_ERRROR)
            break

        try:
            current_user = auth_api.get_current_user()
        except Exception as ex:
            print(f"Failed to refresh current user: {ex}")
            current_user = None

        metadata_manager.sync_from_current_user(current_user)

        online_map = {friend.display_name: friend for friend in online_friends}
        tracked_names = set(activity_status.keys()) | set(online_map.keys()) | set(FRIEND_NAMES)
        for name in tracked_names:
            ensure_activity_record(activity_status, name, name in FRIEND_NAMES)

        user_location_raw = get_presence_location(current_user)
        user_online = bool(user_location_raw and str(user_location_raw).lower() != "offline")

        shared_friend_names = set()

        if user_online:
            for friend_name, friend in online_map.items():
                friend_location = friend.location or "private"
                if friend_location == user_location_raw and friend_location not in ('offline', 'private'):
                    shared_friend_names.add(friend_name)
                    world_instance = worlds_api_instance.get_world_instance(world_id=user_location_raw.split(':')[0], instance_id=user_location_raw.split(':')[1])
                elif ':' in friend_location and ':' in user_location_raw and friend_location.split(':')[0] == user_location_raw.split(':')[0]:
                    print(f"similar world: {friend_name} at {friend_location} while user at {user_location_raw}")
        else:
            await end_active_meets_for_offline_user(activity_status, loop_now_dt, now)
        online_notify_set = {name for name in online_map if activity_status[name]["notify"]}
        online_set = online_notify_set

        for name in sorted(tracked_names):
            record = activity_status[name]
            prev_online = bool(record.get("online", False))
            friend_detail = online_map.get(name)
            currently_online = friend_detail is not None
            should_notify = bool(record.get("notify", name in FRIEND_NAMES))

            if currently_online:
                metadata_manager.update_record_from_detail(name, friend_detail, record)

                location_raw = friend_detail.location or "private"
                location_label = format_location(location_raw, worlds_api_instance, api_client)
                location_type = determine_location_type(location_raw)
                status_description = friend_detail.status_description or ""
                platform_raw = friend_detail.last_platform or ""
                platform_display = normalize_platform_name(platform_raw) or "unknown"

                prev_last_update_str = record.get("last_update")
                prev_last_update_dt = None
                if prev_last_update_str:
                    try:
                        prev_last_update_dt = datetime.strptime(prev_last_update_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        prev_last_update_dt = None

                prev_platform = record.get("last_platform")
                platform_changed = platform_display != prev_platform
                record["last_platform"] = platform_display

                prev_current_location = record.get("current_location", "offline")
                current_location_value = location_raw or "offline"
                current_location_changed = current_location_value != prev_current_location
                record["current_location"] = current_location_value

                prev_status = record.get("status_description")
                status_changed = status_description != prev_status
                record["status_description"] = status_description

                world_name_changed = False
                if location_type not in {"private", "offline"}:
                    target_world_name = location_label or location_raw or record.get("last_location")
                    if target_world_name and target_world_name != record.get("last_location"):
                        record["last_location"] = target_world_name
                        world_name_changed = True

                avatar_name = record.get("last_avatar_name")
                avatar_changed = False
                fetch_avatar = avatar_name is None
                if should_notify and not prev_online:
                    fetch_avatar = True
                if fetch_avatar:
                    new_avatar_name = get_friend_avatar_name(friend_detail, avatars_api_instance, api_client)
                    if new_avatar_name:
                        if new_avatar_name != avatar_name:
                            avatar_changed = True
                        avatar_name = new_avatar_name
                        record["last_avatar_name"] = new_avatar_name
                else:
                    avatar_changed = False

                should_notify_now = should_notify and not prev_online
                if should_notify_now:
                    event_dt = datetime.now()
                    event_now = format_notification_timestamp(event_dt)
                    other_online = sorted(other for other in online_notify_set if other != name)
                    lines = [f"{event_now} {name} came online."]
                    if location_label and location_type not in {"private", "offline"}:
                        lines.append(f"Location: {location_label}")
                    lines.append(f"Platform: {platform_display}")
                    if avatar_name:
                        lines.append(f"Avatar: {avatar_name}")
                    if other_online:
                        lines.append("Online: " + ", ".join(other_online))
                    msg = "\n".join(lines)
                    print(msg)
                    image_url = pick_friend_image_url(friend_detail)
                    image_entries = [{"name": name, "url": image_url}] if image_url else None
                    await send_notification(message=msg, image_entries=image_entries)

                    record["online"] = True
                    record["last_update"] = now
                    record["online_since"] = event_dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    record["online"] = True
                    if platform_changed or current_location_changed or status_changed or avatar_changed or world_name_changed:
                        record["last_update"] = now

                if user_online and name in shared_friend_names and (friend_detail.location or "") == user_location_raw:
                    active = record.get("active_meet")
                    if active and active.get("location") != location_raw:
                        ended_meet = end_meet_session(record, loop_now_dt)
                        if ended_meet:
                            await send_meet_end_notification(name, ended_meet)
                        active = None
                    elif active and prev_last_update_dt and (loop_now_dt - prev_last_update_dt).total_seconds() >= 1800:
                        ended_meet = end_meet_session(record, loop_now_dt)
                        if ended_meet:
                            await send_meet_end_notification(name, ended_meet)
                        active = None
                    others = sorted(other for other in shared_friend_names if other != name)
                    if not record.get("active_meet"):
                        start_meet_session(record, loop_now_dt, location_raw, location_label or location_raw, location_type, others)
                        record["last_update"] = now
                    else:
                        update_meet_session(record, loop_now_dt, others)
                        record["last_update"] = now
                else:
                    if record.get("active_meet"):
                        ended_meet = end_meet_session(record, loop_now_dt)
                        if ended_meet:
                            await send_meet_end_notification(name, ended_meet)

            else:
                if record.get("active_meet"):
                    ended_meet = end_meet_session(record, loop_now_dt)
                    if ended_meet:
                        await send_meet_end_notification(name, ended_meet)
                if prev_online:
                    if should_notify:
                        event_dt = datetime.now()
                        event_now = format_notification_timestamp(event_dt)
                        other_online = sorted(other for other in online_notify_set if other != name)
                        lines = [f"{event_now} {name} went offline."]
                        online_since_str = record.get("online_since")
                        if online_since_str:
                            try:
                                online_since_dt = datetime.strptime(online_since_str, "%Y-%m-%d %H:%M:%S")
                                duration_seconds = (event_dt - online_since_dt).total_seconds()
                                lines.append(f"Online for: {_format_duration_seconds(duration_seconds)}")
                            except Exception:
                                pass
                        if other_online:
                            lines.append("Online: " + ", ".join(other_online))
                        msg = "\n".join(lines)
                        print(msg)
                        log_activity(
                            name,
                            "offline",
                            location="offline",
                            platform=record.get("last_platform", ""),
                            avatar=record.get("last_avatar_name", ""),
                            status="",
                        )
                        await send_notification(msg)
                    record["online"] = False
                    record["last_update"] = now
                    record["current_location"] = "offline"
                    record["status_description"] = ""
                    record["online_since"] = None

        save_activity(activity_status)
        save_cookies(api_client)

        fav_db, next_fav_scan = await handle_favorites_rescan_if_due(api_client, fav_db, args, now, next_fav_scan)

        log_periodic_online_set(now, online_set)
        if now[-7:-3] != "8:00":
            await send_online_digest(now, online_friends, online_set, avatars_api_instance, worlds_api_instance, api_client)

        await asyncio.sleep(CHECK_INTERVAL)

    return fav_db


async def attempt_initial_login(auth_api) -> tuple:
    try:
        current_user = auth_api.get_current_user()
    except UnauthorizedException as e:
        if e.status == 200:
            if "Email 2 Factor Authentication" in e.reason:
                code = input("Enter the 2FA code sent to your email: ").strip()
                auth_api.verify2_fa_email_code(two_factor_email_code=TwoFactorEmailCode(code))
            elif "2 Factor Authentication" in e.reason:
                code = input("Enter 2FA Code: ").strip()
                auth_api.verify2_fa(two_factor_auth_code=TwoFactorAuthCode(code))
            current_user = auth_api.get_current_user()
        elif e.status == 401:
            await send_discord_message(f"UnauthorizedException 401 encountered: {e}", print_to_console=True)
            await sleep_with_retry_after(e, PAUSE_ON_AUTH_ERROR, label="401 Unauthorized.")
            return None, None
        else:
            await send_discord_message(f"UnauthorizedException encountered: {e}", print_to_console=True)
            await sleep_with_retry_after(e, PAUSE_ON_AUTH_ERROR, label="Unauthorized.")
            return None, None
    except ApiException as e:
        if getattr(e, "status", None) == 429:
            await send_discord_message(f"ApiException 429 (rate limit): {e}", print_to_console=True)
            await sleep_with_retry_after(e, PAUSE_ON_AUTH_ERROR, label="Rate limited.")
        else:
            print(f"ApiException when calling API: {e}")
        return None, None
    except ValueError as e:
        print(f"ValueError when calling API: {e}")
        return None, None

    friends_api_instance = FriendsApi(auth_api.api_client)
    return current_user, friends_api_instance


# ---------- NEW: favorites helpers ----------
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _iso_to_date_string(iso_str: str) -> str:
    if not iso_str:
        return ""
    cleaned = iso_str.replace("Z", "+00:00") if iso_str.endswith("Z") else iso_str
    try:
        return datetime.fromisoformat(cleaned).date().isoformat()
    except Exception:
        return ""


def _resolve_avatar_add_date(rec: dict) -> str:
    history = rec.get("history") or []
    for event in reversed(history):
        if (event.get("action") or "").lower() in {"added", "readded"}:
            date_str = _iso_to_date_string(event.get("at"))
            if date_str:
                return date_str
    return _iso_to_date_string(rec.get("first_added_at"))


def load_fav_db():
    if FAV_AVATARS_FILE.exists():
        try:
            with FAV_AVATARS_FILE.open("r", encoding="utf-8") as f:
                db = json.load(f)
                if "avatars" in db:
                    return db
        except Exception:
            pass
    return {"version": 1, "last_scan_at": None, "avatars": {}}


def save_fav_db(db):
    with FAV_AVATARS_FILE.open("w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def _platforms_from_unity_packages(unity_packages):
    plats = set()
    for p in unity_packages or []:
        plat = (p.get("platform") or "").lower()
        if plat:
            plats.add(plat)
    return {
        "platforms": sorted(plats),
        "supports_pc": "standalonewindows" in plats,
        "supports_android": "android" in plats,
    }


def _to_dict(maybe_model):
    if hasattr(maybe_model, "to_dict"):
        return maybe_model.to_dict()
    return maybe_model


def normalize_avatar(raw):
    d = _to_dict(raw) or {}
    # accommodate different casing from SDKs
    unity_packages = d.get("unity_packages") or d.get("unityPackages") or []
    unity_packages = [_to_dict(x) for x in unity_packages]
    platform_info = _platforms_from_unity_packages(unity_packages)

    rec = {
        "id": d.get("id", ""),
        "name": d.get("name", ""),
        "authorId": d.get("author_id") or d.get("authorId") or "",
        "authorName": d.get("author_name") or d.get("authorName") or "",
        "description": d.get("description", ""),
        "imageUrl": d.get("image_url") or d.get("imageUrl") or "",
        "thumbnailImageUrl": d.get("thumbnail_image_url") or d.get("thumbnailImageUrl") or "",
        "releaseStatus": d.get("release_status") or d.get("releaseStatus") or "",
        "tags": d.get("tags") or [],
        "version": d.get("version"),
        **platform_info,
        # tracking
        "active": True,
        "first_added_at": None,
        "last_seen_in_favorites_at": None,
        "removed_at": None,
        "history": [],
        # IMPORTANT: editable map the user will fill via CSV import
        "features": {},  # e.g., {"face_tracking": "yes", "world_drop": "no", "gender": "female"}
    }
    return rec


async def fetch_current_favorite_avatars(api_client: ApiClient, db: dict, force_read: bool = False):
    """
    Returns list of avatar dicts (full records if fetched; DB-cached copies for known IDs when not forced).
    Prefers FavoritesApi pagination + selective hydration so we can skip server calls for known avatars.
    """

    def _to_dict(maybe_model):
        return maybe_model.to_dict() if hasattr(maybe_model, "to_dict") else maybe_model

    # Prefer FavoritesApi path so we can control hydration
    if FavoritesApi is not None and AvatarsApi is not None:
        fav_api = FavoritesApi(api_client)
        av_api = AvatarsApi(api_client)

        # page through favorites (type=avatar)
        favorites = []
        offset = 0
        while True:
            try:
                chunk = fav_api.get_favorites(n=100, offset=offset, type="avatar")
            except TypeError:
                # older SDKs
                chunk = fav_api.get_favorites()
                chunk = [c for c in chunk if (_to_dict(c).get("type") or "").lower() == "avatar"]
            favorites.extend(chunk)
            if len(chunk) < 100:
                break
            offset += 100

        out = []
        avatars_db = (db or {}).get("avatars", {})
        for fav in favorites:
            fd = _to_dict(fav)
            favorite_id = fd.get("target_id") or fd.get("targetId") or fd.get("favorite_id")
            avatar_id = fd.get("id")
            if not favorite_id and not avatar_id:
                continue

            # If cached and not forcing, use DB copy (already normalized)
            if not force_read and favorite_id in avatars_db:
                out.append(avatars_db[favorite_id])
                continue
            if not force_read and avatar_id in avatars_db:
                out.append(avatars_db[avatar_id])
                continue

            # Otherwise hydrate from server
            try:
                av = av_api.get_avatar(favorite_id)
                out.append(_to_dict(av))
                continue
            except Exception as ex:
                cached = avatars_db.get(favorite_id)
                if cached:
                    out.append(cached)
                    continue
            try:
                av = av_api.get_avatar(avatar_id)
                out.append(_to_dict(av))
            except Exception as ex:
                # If server fetch fails but we have a cached copy, fall back to it
                cached = avatars_db.get(avatar_id)
                if cached:
                    out.append(cached)
                    continue

        return out

    # If we only have AvatarsApi.get_favorite_avatars (can’t skip per-avatar calls)
    if AvatarsApi is not None and hasattr(AvatarsApi, "get_favorite_avatars"):
        api = AvatarsApi(api_client)
        favs = api.get_favorite_avatars()
        return [_to_dict(a) for a in favs]

    # Raw REST fallback
    resp = api_client.call_api(
        "/avatars/favorites", "GET",
        response_type="object",
        auth_settings=[],
        _return_http_data_only=True
    )
    return resp if isinstance(resp, list) else []


def _derive_image_ext(url: str) -> str:
    try:
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp"}:
            return ext
    except Exception:
        pass
    return ".jpg"


def _sanitize_avatar_name(name: str) -> str:
    if not name:
        return ""
    try:
        # drop non-ascii
        ascii_only = name.encode("ascii", errors="ignore").decode("ascii")
        # allow common safe filename chars; replace others with '_'
        safe_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ._-()[]{}")
        cleaned = ''.join(c if c in safe_chars else '_' for c in ascii_only)
        # collapse whitespace
        cleaned = ' '.join(cleaned.split())
        # trim to 30 chars
        return cleaned[:30]
    except Exception:
        return ""


def _find_existing_image_path(avatar_id: str):
    try:
        candidates = []
        for p in AVATAR_IMG_DIR.glob(f"{avatar_id}*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                candidates.append(p)
        if not candidates:
            return None
        # pick most recently modified
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    except Exception:
        return None


def cache_avatar_image(avatar_id: str, image_url: str, avatar_name: str = None):
    if not image_url:
        return
    try:
        desired_name_part = _sanitize_avatar_name(avatar_name or "")
        # First, compute the desired output path based on the URL's extension
        url_ext = _derive_image_ext(image_url)
        out_path = AVATAR_IMG_DIR / (
            f"{avatar_id}_{desired_name_part}{url_ext}" if desired_name_part else f"{avatar_id}{url_ext}"
        )
        # Early-exit: if the exact desired file already exists, it's cached
        if out_path.exists():
            return

        # Otherwise, look for any existing cached file by ID prefix and rename if needed
        existing = _find_existing_image_path(avatar_id)
        if existing is not None:
            # Ensure filename prefix + name correctness; keep current ext
            ext = existing.suffix
            desired_filename = (
                f"{avatar_id}_{desired_name_part}{ext}" if desired_name_part else f"{avatar_id}{ext}"
            )
            if existing.name != desired_filename:
                target = AVATAR_IMG_DIR / desired_filename
                try:
                    if target.exists() and target != existing:
                        target.unlink()
                    existing.rename(target)
                except Exception as re:
                    print(f"[image-cache] rename failed for {avatar_id}: {re}")
            return  # file already present (possibly renamed)

            # Note: if rename failed and we fell through, we would download, but we return above.

        # No existing file; download and save with desired filename
        resp = _HTTP.get(image_url, timeout=20, allow_redirects=True)
        if resp.status_code == 200 and resp.content:
            out_path.write_bytes(resp.content)
        else:
            print(f"[image-cache] {avatar_id}: HTTP {resp.status_code} {resp.text[:160]}")
    except Exception as e:
        print(f"[image-cache] {avatar_id}: {e}")


def _extract_world_id(location: str):
    if not location:
        return None
    location = location.strip()
    if not location.startswith("wrld_"):
        return None
    if ":" in location:
        return location.split(":", 1)[0]
    return location


def _lookup_world_name(world_id: str, worlds_api, api_client):
    if not world_id:
        return None
    if world_id in WORLD_NAME_CACHE:
        return WORLD_NAME_CACHE[world_id]

    name = None
    try:
        if worlds_api is not None:
            world = worlds_api.get_world(world_id)
            name = getattr(world, "name", None)
            if not name and hasattr(world, "to_dict"):
                world_dict = world.to_dict()
                if isinstance(world_dict, dict):
                    name = world_dict.get("name")
    except Exception as ex:
        print(f"World lookup failed for {world_id}: {ex}")

    WORLD_NAME_CACHE[world_id] = name
    return name


def format_location(location: str, worlds_api=None, api_client=None):
    if not location:
        return "unknown"
    world_id = _extract_world_id(location)
    if not world_id:
        return location

    world_name = _lookup_world_name(world_id, worlds_api, api_client)
    return world_name or world_id


def resolve_world_name_only(location: str, worlds_api=None, api_client=None):
    if not location:
        return None
    world_id = _extract_world_id(location)
    if not world_id:
        return None
    return _lookup_world_name(world_id, worlds_api, api_client)


FRIEND_IMAGE_ATTR_PREFS = [
    ("profile_pic_override", "profilePicOverride"),
    ("current_avatar_thumbnail_image_url", "currentAvatarThumbnailImageUrl"),
    ("user_icon", "userIcon"),
    ("current_avatar_image_url", "currentAvatarImageUrl"),
]


def _extract_friend_attr(friend, attr_names):
    for attr in attr_names:
        if isinstance(friend, dict):
            val = friend.get(attr)
        else:
            val = getattr(friend, attr, None)
        if val:
            return val
    return None


def pick_friend_image_url(friend):
    if friend is None:
        return None
    for names in FRIEND_IMAGE_ATTR_PREFS:
        url = _extract_friend_attr(friend, names if isinstance(names, (list, tuple)) else (names,))
        if url:
            return url
    return None


def _extract_avatar_id(friend):
    if not friend:
        return None
    possible_keys = [
        "current_avatar",
        "currentAvatar",
        "current_avatar_id",
        "currentAvatarId",
        "avatar_id",
        "avatarId",
    ]
    if isinstance(friend, dict):
        for key in possible_keys:
            val = friend.get(key)
            if val:
                return val
        return None
    for key in possible_keys:
        val = getattr(friend, key, None)
        if val:
            return val
    return None


def _lookup_avatar_name(avatar_id: str, avatars_api, api_client):
    if not avatar_id:
        return None
    if avatar_id in AVATAR_NAME_CACHE:
        return AVATAR_NAME_CACHE[avatar_id]

    name = None
    try:
        if avatars_api is not None:
            avatar = avatars_api.get_avatar(avatar_id)
            name = getattr(avatar, "name", None)
            if not name and hasattr(avatar, "to_dict"):
                avatar_dict = avatar.to_dict()
                if isinstance(avatar_dict, dict):
                    name = avatar_dict.get("name")
    except Exception as ex:
        print(f"Avatar lookup failed for {avatar_id}: {ex}")

    AVATAR_NAME_CACHE[avatar_id] = name
    return name


def get_friend_avatar_name(friend, avatars_api=None, api_client=None):
    avatar_id = None
    if friend is not None:
        avatar_id = _extract_avatar_id(friend)
    if not avatar_id:
        return None
    return _lookup_avatar_name(avatar_id, avatars_api, api_client)


def merge_favorites_snapshot(db, avatars):
    now = now_iso()
    current_ids = set()

    for av in avatars:
        norm = normalize_avatar(av)
        aid = norm["id"]
        if not aid:
            continue
        current_ids.add(aid)
        if aid not in db["avatars"]:
            # New avatar added
            norm["first_added_at"] = now
            norm["last_seen_in_favorites_at"] = now  # set ONLY on first add
            norm["active"] = True
            norm["history"].append({"at": now, "action": "added"})
            db["avatars"][aid] = norm
            # cache image on first add
            cache_avatar_image(aid, norm.get("imageUrl"), norm.get("name"))
        else:
            rec = db["avatars"][aid]
            # keep existing features/history; update metadata (do NOT touch last_seen_in_favorites_at here)
            prev_image_url = rec.get("imageUrl")
            rec.update({
                "name": norm["name"],
                "authorId": norm["authorId"],
                "authorName": norm["authorName"],
                "description": norm["description"],
                "imageUrl": norm["imageUrl"],
                "thumbnailImageUrl": norm["thumbnailImageUrl"],
                "releaseStatus": norm["releaseStatus"],
                "tags": norm["tags"],
                "version": norm["version"],
                "supports_pc": norm["supports_pc"],
                "supports_android": norm["supports_android"],
                "platforms": norm["platforms"],
            })
            # resurrect logic does NOT change last_seen_in_favorites_at
            if not rec.get("active", True):
                rec["active"] = True
                rec["removed_at"] = None
                rec["history"].append({"at": now, "action": "readded"})
            # ensure cached image exists and filename includes current name
            if rec.get("imageUrl") and rec.get("id"):
                cache_avatar_image(rec["id"], rec["imageUrl"], rec.get("name"))
            # DO NOT set rec["last_seen_in_favorites_at"] here

    # mark removals
    for aid, rec in list(db["avatars"].items()):
        if rec.get("active", False) and aid not in current_ids:
            rec["active"] = False
            rec["removed_at"] = now
            rec["last_seen_in_favorites_at"] = now  # set ONLY when disabled/removed
            rec["history"].append({"at": now, "action": "removed"})

    db["last_scan_at"] = now
    return db


def export_avatars_csv(db_path: Path, csv_path: Path):
    db = load_fav_db() if db_path is None else json.loads(Path(db_path).read_text(encoding="utf-8"))
    avatars = db.get("avatars", {})

    base_cols = ["id", "name", "active", "supports_android"]
    fieldnames = base_cols + FAV_AVATAR_FEATURES

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Sort by name (case-insensitive), then by id for stability
        sorted_recs = sorted(
            avatars.values(),
            key=lambda r: ((r.get("name") or "").lower(), r.get("id") or "")
        )
        for rec in sorted_recs:
            row = {
                "id": rec.get("id"),
                "name": rec.get("name"),
                "active": rec.get("active"),
                "supports_android": rec.get("supports_android"),
            }
            feats = rec.get("features", {}) or {}
            for key in FAV_AVATAR_FEATURES:
                # write empty string if missing
                value = feats.get(key, "")
                if key == "add_date" and not value and rec.get("active") and not rec.get("removed_at"):
                    value = _resolve_avatar_add_date(rec)
                row[key] = value
            w.writerow(row)


def import_avatars_csv(csv_path: Path):
    db = load_fav_db()
    avatars = db.get("avatars", {})

    updated = 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # sanity: ensure required columns exist
        missing = [c for c in ["id", "name", "active", "supports_android"] if c not in r.fieldnames]
        if missing:
            print(f"[WARN] CSV missing columns: {missing} (continuing; will import feature columns only)")
        for row in r:
            aid = (row.get("id") or "").strip()
            if not aid or aid not in avatars:
                continue

            rec = avatars[aid]
            # build features map from feature columns only
            feats = dict(rec.get("features", {}) or {})
            changed = False
            for key in FAV_AVATAR_FEATURES:
                if key in row:
                    val = row[key]
                    # store as-is (string), strip harmlessly
                    val = "" if val is None else str(val)
                    if feats.get(key, "") != val:
                        feats[key] = val
                        changed = True
            if changed:
                rec["features"] = feats
                updated += 1

    save_fav_db(db)
    print(f"Imported features for {updated} avatars into {FAV_AVATARS_FILE}")


# ---------- Main monitor with hourly favorites scan ----------
async def main_loop(args):
    activity_status = load_activity()
    config = Configuration(username=EMAIL if EMAIL else USERNAME, password=PASSWORD)
    print(f"Using User-Agent: {USER_AGENT}")

    fav_db = load_fav_db()

    with ApiClient(config) as api_client:
        api_client.user_agent = USER_AGENT

        if getattr(args, "clean_login", False):
            clear_saved_cookies()
            print("Clean login requested; skipping cached cookies.")
        else:
            load_cookies(api_client)

        auth_api = AuthenticationApi(api_client)
        try:
            current_user, friends_api_instance = await attempt_initial_login(auth_api)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("initial get_current_user timed out after 5 seconds") from exc

        if not current_user or not friends_api_instance:
            return

        print(f"Logged in as: {current_user.display_name}")
        save_cookies(api_client)

        avatars_api_instance, worlds_api_instance, users_api_instance = init_optional_apis(api_client)
        metadata_manager = FriendMetadataManager(users_api_instance, activity_status)

        if metadata_manager.sync_from_current_user(current_user):
            save_activity(activity_status)

        try:
            online_friends = friends_api_instance.get_friends(offline=False, offset=0)
        except Exception as ex:
            print(f"Failed to fetch initial friends list: {ex}")
            online_friends = []

        print("Unfiltered online: " + ", ".join([f.display_name for f in online_friends]))
        initial_online = [f for f in online_friends if f.display_name in FRIEND_NAMES]
        await handle_initial_online_snapshot(
            initial_online,
            activity_status,
            metadata_manager,
            avatars_api_instance,
            worlds_api_instance,
            api_client,
        )

        fav_db = await monitor_loop(
            args,
            api_client,
            auth_api,
            friends_api_instance,
            avatars_api_instance,
            worlds_api_instance,
            metadata_manager,
            activity_status,
            fav_db,
        )


# ---------- CLI for export/import only ----------
def parse_args():
    p = argparse.ArgumentParser(description="VRC friend monitor + favorite avatars tracker")
    p.add_argument("--export-avatars", metavar="CSV_PATH", help="Export favorite avatars to CSV and exit")
    p.add_argument("--import-avatars", metavar="CSV_PATH", help="Import features from CSV into favorite_avatars.json and exit")
    p.add_argument("--force-read-avatars", action="store_true",
                   help="Force re-read full avatar details from server even if already in DB")
    p.add_argument("--clean-login", action="store_true", default=False,
                   help="Skip cached cookies and perform a clean login on startup")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.export_avatars:
        export_avatars_csv(None, Path(args.export_avatars))
        print(f"Exported avatars to {args.export_avatars}")
    elif args.import_avatars:
        import_avatars_csv(Path(args.import_avatars))
    else:
        while True:
            asyncio.run(main_loop(args))
