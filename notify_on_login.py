import asyncio
import aiohttp
import json
import os
import requests

# from discord import Webhook, SyncWebhook
# from discord_webhook import DiscordWebhook
import csv
from http.cookiejar import Cookie
from dotenv import load_dotenv
from vrchatapi import ApiClient
from vrchatapi.configuration import Configuration
from vrchatapi import AuthenticationApi, FriendsApi
from vrchatapi.exceptions import UnauthorizedException, ApiException
from vrchatapi.models.two_factor_auth_code import TwoFactorAuthCode
from vrchatapi.models.two_factor_email_code import TwoFactorEmailCode
from datetime import datetime
from pathlib import Path

load_dotenv(".env")

# Credentials / contact info
USERNAME = os.getenv("VRCHAT_USERNAME")
PASSWORD = os.getenv("VRCHAT_PASSWORD")
EMAIL = os.getenv("VRCHAT_EMAIL")
contact_info = EMAIL if EMAIL else "no-contact@example.com"

# Notifications configuration
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Friend list to monitor (comma separated in env)
VRC_FRIEND_NAMES = os.getenv("VRC_FRIEND_NAMES", "")
FRIEND_NAMES = [name.strip() for name in VRC_FRIEND_NAMES.split(",") if name.strip()]

CHECK_INTERVAL = 60  # seconds

# Files to keep activity and cookie data
STATE_FILE = Path("friend_activity.json")
CSV_LOG_FILE = Path("user_activity_log.csv")
COOKIES_FILE = Path("auth_cookies.json")


def load_activity():
    """
    Load the JSON configuration tracking friend activity.
        Structure: { "Alice": {"online": bool, "last_update": timestamp_str}, ... }
    """
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    # Ensure all friend names are present
    for name in FRIEND_NAMES:
        if name not in data:
            data[name] = {"online": False, "last_update": None}
    return data


def save_activity(activity):
    with STATE_FILE.open("w") as f:
        json.dump(activity, f, indent=4)


def log_activity(user_name: str, event: str):
    """
    Append an entry to the CSV log file.
        Each row: timestamp, user_name, event (online/offline)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = CSV_LOG_FILE.exists()
    with CSV_LOG_FILE.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "user", "event"])
        writer.writerow([now, user_name, event])


async def send_discord_message(message: str, image_url: str = None):
    if not WEBHOOK_URL:
        return
    # discord.webhook = requests.post(WEBHOOK_URL, json={"content": message})

    if image_url is not None:
        try:
            response = requests.get(image_url)
            image = response.content
        except requests.RequestException as e:
            data = {"content": message}
            resp = requests.post(WEBHOOK_URL, json=data)
            return
        data = {"payload_json": (None, f'{{"content": "{message}"}}'), "media.png": image}
        resp = requests.post(WEBHOOK_URL, files=data)
    else:
        data = {"content": message}
        resp = requests.post(WEBHOOK_URL, json=data)
    print(f"Discord webhook status: {resp.content} - {resp.status_code}")


async def send_telegram_message(message: str, image_url: str = None):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)


async def send_notification(message: str, image_url: str = None):
    """Send notifications via Discord and Telegram."""
    await send_discord_message(message, image_url)
    await send_telegram_message(message, image_url)


def make_cookie(name, value):
    return Cookie(0, name, value,
                  None, False,
                  "api.vrchat.cloud", True, False,
                  "/", False,
                  False,
                  173106866300,
                  False,
                  None,
                  None, {})


def load_cookies(api_client: ApiClient):
    """Load cookies from disk into the session's cookie jar."""
    if COOKIES_FILE.exists():
        try:
            with COOKIES_FILE.open("r") as f:
                cookies = json.load(f)
                api_client.rest_client.cookie_jar.set_cookie(make_cookie("auth", cookies.get("auth", None)))
                api_client.rest_client.cookie_jar.set_cookie(make_cookie("twoFactorAuth", cookies.get("twoFactorAuth", None)))

            # cookie = cookies.get("cookie", None)
            print("Loaded auth cookie from disk.")

        except Exception as e:
            print(f"Error loading cookies: {e}")


def save_cookies(api_client: ApiClient):
    """Save cookies from the session's cookie jar to disk."""
    cookie_jar = api_client.rest_client.cookie_jar._cookies["api.vrchat.cloud"]["/"]
    # print("auth: " + cookie_jar["auth"].value)
    # print("twoFactorAuth: " + cookie_jar["twoFactorAuth"].value)
    with COOKIES_FILE.open("w") as f:
        json.dump(
            {
                "auth": cookie_jar["auth"].value,
                "twoFactorAuth": cookie_jar["twoFactorAuth"].value,
            },
            f
        )
    print("Saved auth cookie to disk.")


async def main():
    # await send_discord_message(message="Hello from VRCNotify!")  # , image_url="https://api.vrchat.cloud/api/1/file/file_30fcc98f-bc1a-412b-9368-ce54b10a0c8a/1")
    # return
    activity_status = load_activity()

    # If an email is provided, use it for login; otherwise, use the username.
    config = Configuration(username=EMAIL if EMAIL else USERNAME, password=PASSWORD)
    print(f"Using User-Agent: VRCNotify/1.0 {contact_info}")

    # Instantiate ApiClient with the saved cookie if any, as a context manager
    with ApiClient(config) as api_client:
        api_client.user_agent = f"VRCNotify/1.0 {contact_info}"
        # Optionally, load cookies again (redundant if already set)
        load_cookies(api_client)

        auth_api = AuthenticationApi(api_client)
        try:
            # Calling getCurrentUser on Authentication API logs you in if the user isn't already logged in.
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
                # Retry getting current user after 2FA verification
                current_user = auth_api.get_current_user()
                friends_api_instance = FriendsApi(api_client)
            else:
                print(f"UnauthorizedException encountered: {e}")
                return
        except ApiException as e:
            print(f"ApiException when calling API: {e}")
            return

        print(f"Logged in as: {current_user.display_name}")
        # Save cookies after a successful login so next run reuses them
        save_cookies(api_client)

        # Initial check: send a combined message if any monitored friends are online.
        online_friends = friends_api_instance.get_friends()
        initial_online = [f.display_name for f in online_friends if f.display_name in FRIEND_NAMES]
        if initial_online:
            msg = "Initial online friends: " + ", ".join(initial_online)
            print(msg)
            await send_notification(msg)
            now_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for name in initial_online:
                activity_status[name] = {"online": True, "last_update": now_timestamp}
                log_activity(name, "offline")
            save_activity(activity_status)

        # Main loop: periodically check friend status
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            online_friends = friends_api_instance.get_friends()
            # Build set of names that are online from our friend list.
            online_set = {f.display_name for f in online_friends if f.display_name in FRIEND_NAMES}

            # Check each monitored friend for status changes.
            for name in FRIEND_NAMES:
                prev_online = activity_status.get(name, {"online": False})["online"]
                currently_online = name in online_set

                if currently_online and not prev_online:
                    # Friend just went online.
                    # Retrieve additional info if available.
                    friend_detail = next((f for f in online_friends if f.display_name == name), None)
                    avatar_image = None
                    if friend_detail:
                        location = friend_detail.location or "private"
                        # avatar_image = friend_detail.current_avatar_image_url
                        last_platform = f"({friend_detail.last_platform})" if friend_detail.last_platform else ""
                        instance_id = location if location else "unknown"
                        msg = f"{now} [{name}] just logged in!\n" f"World: **{last_platform}**\n" f"Instance: `{instance_id}`"
                        print(msg)
                        log_activity(name, "online")
                        await send_notification(message=msg, image_url=avatar_image)
                        activity_status[name] = {"online": True, "last_update": now}
                elif not currently_online and prev_online:
                    # Friend just went offline.
                    msg = f"{now} [{name}] went offline."
                    print(msg)
                    log_activity(name, "offline")
                    await send_notification(msg)
                    activity_status[name] = {
                        "online": False,
                        "last_update": now,
                        "last_platform": last_platform,
                        "last_location": instance_id,
                    }

            save_activity(activity_status)
            # Optionally, save cookies periodically
            save_cookies(api_client)
            await asyncio.sleep(CHECK_INTERVAL)



if __name__ == "__main__":
    asyncio.run(main())
