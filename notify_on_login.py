import asyncio
import time
import argparse
import json
import os
import random
import requests
import csv
from http.cookiejar import Cookie
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
import requests
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
]

# NEW: avatar image cache directory
AVATAR_IMG_DIR = Path("avatar_images")
AVATAR_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Optional typed APIs (guarded)
try:
    from vrchatapi import AvatarsApi, FavoritesApi, WorldsApi
except Exception:
    AvatarsApi = None
    FavoritesApi = None
    WorldsApi = None

# cache resolved world and avatar names to limit API calls
WORLD_NAME_CACHE = {}
AVATAR_NAME_CACHE = {}


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


# ---------- Friend-activity persistence (unchanged) ----------
def load_activity():
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    for name in FRIEND_NAMES:
        if name not in data:
            data[name] = {"online": False, "last_update": None}
    return data


def save_activity(activity):
    with STATE_FILE.open("w") as f:
        json.dump(activity, f, indent=4)


def log_activity(user_name: str, event: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = CSV_LOG_FILE.exists()
    with CSV_LOG_FILE.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "user", "event"])
        writer.writerow([now, user_name, event])


# single session with required UA for VRChat CDN
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": USER_AGENT})


async def send_discord_message(message: str, image_url: str = None, image_urls=None, image_entries=None):
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


# ---------- NEW: favorites helpers ----------
def now_iso():
    return datetime.now(timezone.utc).isoformat()


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
                row[key] = feats.get(key, "")
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
async def main_loop(args, first_run=False):
    activity_status = load_activity()
    config = Configuration(username=EMAIL if EMAIL else USERNAME, password=PASSWORD)
    print(f"Using User-Agent: {USER_AGENT}")

    fav_db = load_fav_db()

    with ApiClient(config) as api_client:
        api_client.user_agent = USER_AGENT
        load_cookies(api_client)

        auth_api = AuthenticationApi(api_client)
        try:
            current_user = auth_api.get_current_user()
            friends_api_instance = FriendsApi(api_client)
        except UnauthorizedException as e:
            if e.status == 200:
                if "Email 2 Factor Authentication" in e.reason:
                    code = input("Enter the 2FA code sent to your email: ").strip()
                    auth_api.verify2_fa_email_code(two_factor_email_code=TwoFactorEmailCode(code))
                elif "2 Factor Authentication" in e.reason:
                    code = input("Enter 2FA Code: ").strip()
                    auth_api.verify2_fa(two_factor_auth_code=TwoFactorAuthCode(code))
                current_user = auth_api.get_current_user()
                friends_api_instance = FriendsApi(api_client)
            else:
                print(f"UnauthorizedException encountered: {e}")
                return
        except ApiException as e:
            print(f"ApiException when calling API: {e}")
            return
        except ValueError as e:
            print(f"ValueError when calling API: {e}")
            return

        print(f"Logged in as: {current_user.display_name}")
        save_cookies(api_client)

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

        # Initial favorites scan
        try:
            current_favs = await fetch_current_favorite_avatars(api_client, fav_db, force_read=args.force_read_avatars)
            fav_db = merge_favorites_snapshot(fav_db, current_favs)
            save_fav_db(fav_db)
            print(f"Favorites scan complete: {len(current_favs)} favorites. DB total avatars: {len(fav_db['avatars'])}")
        except Exception as ex:
            print(f"Favorite avatar scan failed: {ex}")

        # Initial friend summary (unchanged)
        online_friends = friends_api_instance.get_friends(offline=False, offset=0)
        # Print unfiltered list of online friends
        print("Unfiltered online: " + ", ".join([f.display_name for f in online_friends]))
        initial_online = [f for f in online_friends if f.display_name in FRIEND_NAMES]
        if initial_online:
            initial_online_names = []
            initial_image_entries = []
            for f in initial_online:
                raw_location = f.location or "private"
                platform_raw = f.last_platform or "unknown"
                plat = normalize_platform_name(platform_raw) or "unknown"
                formatted_location = format_location(raw_location, worlds_api_instance, api_client)
                avatar_name = get_friend_avatar_name(f, avatars_api_instance, api_client)
                avatar_suffix = f" • {avatar_name}" if avatar_name else ""
                initial_online_names.append(f"{f.display_name}({plat}/{formatted_location}{avatar_suffix})")
                image_url = pick_friend_image_url(f)
                if image_url:
                    initial_image_entries.append({"name": f.display_name, "url": image_url})
            msg = "Initial online friends: " + ", ".join(initial_online_names)
            print(msg)
            await send_notification(msg, image_entries=initial_image_entries)
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for f in initial_online:
                name = f.display_name
                raw_location = f.location or "private"
                platform_raw = f.last_platform or "unknown"
                plat = normalize_platform_name(platform_raw) or "unknown"
                formatted_location = format_location(raw_location, worlds_api_instance, api_client)
                avatar_name = get_friend_avatar_name(f, avatars_api_instance, api_client)
                avatar_log_suffix = f" Avatar: {avatar_name}" if avatar_name else ""
                activity_status[name] = {
                    "online": True,
                    "last_update": now_ts,
                    "last_platform": f.last_platform,
                    "last_location": raw_location,
                    "status_description": getattr(f, "status_description", "") or "",
                    "last_avatar_name": avatar_name,
                }
                log_activity(name, "online")
                log_activity(name, f"online. Platform: **{plat}** Location: `{formatted_location}`{avatar_log_suffix}")
            save_activity(activity_status)

        # Schedule hourly favorites re-scan
        def next_hour_epoch():
            now = datetime.now()
            return time.time() + ((60 - now.minute - 1) * 60 + (60 - now.second))

        next_fav_scan = next_hour_epoch()

        # Main loop (friend monitoring + hourly favorites)
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                online_friends = friends_api_instance.get_friends()
            except Exception as ex:
                await send_notification(message=f"Critical error: {ex}, waiting 1 hour.")
                await asyncio.sleep(PAUSE_ON_CRITICAL_ERRROR)
                break

            online_set = {f.display_name for f in online_friends if f.display_name in FRIEND_NAMES}

            for name in FRIEND_NAMES:
                prev_online = activity_status.get(name, {"online": False})["online"]
                currently_online = name in online_set

                if currently_online and not prev_online:
                    friend_detail = next((f for f in online_friends if f.display_name == name), None)
                    avatar_image = None
                    if friend_detail:
                        location = friend_detail.location or "private"
                        formatted_location = format_location(location, worlds_api_instance, api_client)
                        status_description = friend_detail.status_description or ""
                        platform_raw = friend_detail.last_platform or ""
                        platform_display = normalize_platform_name(platform_raw) or "unknown"
                        last_platform = f"({platform_display})"
                        avatar_name = get_friend_avatar_name(friend_detail, avatars_api_instance, api_client)
                        instance_id = formatted_location if formatted_location else (location or "unknown")
                        is_private_instance = instance_id.lower() == "private"
                        lines = [
                            f"{now} [{name}] just logged in!",
                            f"Platform: **{last_platform}**",
                        ]
                        if not is_private_instance:
                            lines.append(f"Instance: `{instance_id}`")
                        image_url = pick_friend_image_url(friend_detail)
                        if image_url:
                            avatar_image = image_url
                        if avatar_name:
                            lines.append(f"Avatar: {avatar_name}")
                        lines.append(f"Status: {status_description}")
                        msg = "\n".join(lines)
                        print(msg)
                        avatar_log_suffix = f" Avatar: {avatar_name}" if avatar_name else ""
                        instance_log_suffix = f" Instance: `{instance_id}`" if not is_private_instance else ""
                        log_activity(name, f"online. Platform: **{last_platform}**{instance_log_suffix}{avatar_log_suffix} Status: {status_description}")
                        image_entries = [{"name": name, "url": avatar_image}] if avatar_image else None
                        await send_notification(message=msg, image_entries=image_entries)
                        activity_status[name] = {
                            "online": True,
                            "last_update": now,
                            "last_platform": platform_display,
                            "last_location": location,
                            "status_description": status_description,
                            "last_avatar_name": avatar_name,
                        }
                elif not currently_online and prev_online:
                    other_online = sorted(online_set)
                    lines = [f"{now} [{name}] went offline."]
                    if other_online:
                        lines.append("Online: " + ", ".join(other_online))
                    msg = "\n".join(lines)
                    print(msg)
                    log_activity(name, "offline")
                    await send_notification(msg)
                    if activity_status.get(name):
                        activity_status[name]["online"] = False
                        activity_status[name]["last_update"] = now
                    else:
                        activity_status[name] = {"online": False, "last_update": now}

            save_activity(activity_status)
            save_cookies(api_client)

            # Hourly favorites re-scan
            if time.time() >= next_fav_scan:
                try:
                    # use current DB (avoid re-reading unchanged avatars)
                    fav_db_current = load_fav_db()
                    current_favs = await fetch_current_favorite_avatars(api_client, fav_db_current, force_read=args.force_read_avatars)

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
                        await send_notification("Favorites changed:\n" + "\n".join(lines))
                    else:
                        print(f"{now} Favorites scan: no changes.")
                except Exception as ex:
                    print(f"Hourly favorites scan failed: {ex}")

                next_fav_scan = next_hour_epoch()

            if now[-4] == "0":
                # Print names of those online
                print(f"{now} {','.join(online_set)}")
            if now[-7:-3] == "8:00" or first_run:
                # Twice a day, print who's online, with world names when available
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
                
                message=f"{now} Online:  {', '.join(online_entries)}"
                if others_online:
                    message += f"\nOthers:  {', '.join(others_online)}"

                await send_notification(message=message, image_entries=digest_image_entries)

            await asyncio.sleep(CHECK_INTERVAL)


# ---------- CLI for export/import only ----------
def parse_args():
    p = argparse.ArgumentParser(description="VRC friend monitor + favorite avatars tracker")
    p.add_argument("--export-avatars", metavar="CSV_PATH", help="Export favorite avatars to CSV and exit")
    p.add_argument("--import-avatars", metavar="CSV_PATH", help="Import features from CSV into favorite_avatars.json and exit")
    p.add_argument("--force-read-avatars", action="store_true",
                   help="Force re-read full avatar details from server even if already in DB")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.export_avatars:
        export_avatars_csv(None, Path(args.export_avatars))
        print(f"Exported avatars to {args.export_avatars}")
    elif args.import_avatars:
        import_avatars_csv(Path(args.import_avatars))
    else:
        first_run=True
        while True:
            asyncio.run(main_loop(args, first_run=first_run))
            first_run=False
