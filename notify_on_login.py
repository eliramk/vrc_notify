import asyncio
import aiohttp
import json
import os
import csv
from dotenv import load_dotenv
from vrchatapi import ApiClient
from vrchatapi.configuration import Configuration
from vrchatapi import AuthenticationApi, FriendsApi
from vrchatapi.exceptions import UnauthorizedException, ApiException
from vrchatapi.models.two_factor_auth_code import TwoFactorAuthCode
from vrchatapi.models.two_factor_email_code import TwoFactorEmailCode
from datetime import datetime
from pathlib import Path

load_dotenv()

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
FRIEND_NAMES = [name.strip() for name in VRC_FRIEND_NAMES.split(',') if name.strip()]

CHECK_INTERVAL = 60  # seconds

# Files to keep activity data
STATE_FILE = Path("friend_activity.json")
CSV_LOG_FILE = Path("user_activity_log.csv")


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


async def send_discord_message(message: str):
    if not WEBHOOK_URL:
        return
    async with aiohttp.ClientSession() as session:
        await session.post(WEBHOOK_URL, json={"content": message})


async def send_telegram_message(message: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)


async def send_notification(message: str):
    """Send notifications via Discord and Telegram."""
    await send_discord_message(message)
    await send_telegram_message(message)


async def main():
    activity_status = load_activity()

    # If an email is provided, use it for login; otherwise, use the username.
    config = Configuration(username=EMAIL if EMAIL else USERNAME, password=PASSWORD)

    print(f"Using User-Agent: VRCNotify/1.0 {contact_info}")

    with ApiClient(config) as api_client:
        api_client.user_agent = f"VRCNotify/1.0 {contact_info}"

        auth_api = AuthenticationApi(api_client)

        try:
            # Step 3. Calling getCurrentUser on Authentication API logs you in if the user isn't already logged in.
            current_user = await auth_api.get_current_user()
            friends_api_instance = FriendsApi(api_client)
        except UnauthorizedException as e:
            if e.status == 200:
                if "Email 2 Factor Authentication" in e.reason:
                    # Step 3.5. Calling email verify2fa if the account has 2FA disabled
                    await auth_api.verify2_fa_email_code(two_factor_email_code=TwoFactorEmailCode(input("Enter the 2FA Code sent to your email: ")))
                elif "2 Factor Authentication" in e.reason:
                    # Step 3.5. Calling verify2fa if the account has 2FA enabled
                    await auth_api.verify2_fa(two_factor_auth_code=TwoFactorAuthCode(input("2FA Code: ")))
                current_user = await auth_api.get_current_user()
                friends_api_instance = FriendsApi(api_client)
            else:
                print("Exception when calling API: %s", e)
        except ApiException as e:
            print("Exception when calling API: %s", e)

            print(f"Logged in as: {current_user.display_name}")

        # Initial check: if any monitored friends are online, send one combined message.
        online_friends = await friends_api_instance.get_friends()
        initial_online = [f.display_name for f in online_friends if f.display_name in FRIEND_NAMES]
        if initial_online:
            msg = "Initial online friends: " + ", ".join(initial_online)
            print(msg)
            await send_notification(msg)
            now_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for name in initial_online:
                activity_status[name] = {"online": True, "last_update": now_timestamp}
            save_activity(activity_status)

        # Main loop: periodically check status.
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            online_friends = await friends_api_instance.get_friends()
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
                    if friend_detail:
                        location = friend_detail.location or "private"
                        world_name = friend_detail.world.name if friend_detail.world else "Unknown World"
                        instance_id = location if location else "unknown"
                        msg = (
                            f"[{now}] **{name}** just logged in!\n"
                            f"World: **{world_name}**\n"
                            f"Instance: `{instance_id}`"
                        )
                        print(msg)
                        await send_notification(msg)
                        log_activity(name, "online")
                        activity_status[name] = {"online": True, "last_update": now}

                elif not currently_online and prev_online:
                    # Friend just went offline.
                    msg = f"[{now}] **{name}** went offline."
                    print(msg)
                    await send_notification(msg)
                    log_activity(name, "offline")
                    activity_status[name] = {"online": False, "last_update": now}

            save_activity(activity_status)
            await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
