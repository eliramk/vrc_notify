import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv
from vrchatapi import VRChatAPI
from vrchatapi.models import Configuration
from vrchatapi.api import authentication_api, friends_api
from datetime import datetime
from pathlib import Path

load_dotenv()

USERNAME = os.getenv("VRCHAT_USERNAME")
PASSWORD = os.getenv("VRCHAT_PASSWORD")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
VRC_FRIEND_NAMES = os.getenv("VRC_FRIEND_NAMES")
FRIEND_NAMES = VRC_FRIEND_NAMES.split(',')
CHECK_INTERVAL = 60  # seconds
STATE_FILE = Path("friend_status.json")


def load_last_status():
    if STATE_FILE.exists():
        with STATE_FILE.open("r") as f:
            return json.load(f)
    return {name: False for name in FRIEND_NAMES}


def save_last_status(status):
    with STATE_FILE.open("w") as f:
        json.dump(status, f)


async def send_discord_message(message: str):
    async with aiohttp.ClientSession() as session:
        await session.post(WEBHOOK_URL, json={"content": message})


async def main():
    last_status = load_last_status()

    config = Configuration(username=USERNAME, password=PASSWORD)
    async with VRChatAPI(config) as api:
        auth_api = authentication_api.AuthenticationApi(api.api_client)
        friends = friends_api.FriendsApi(api.api_client)

        user = await auth_api.get_current_user()
        print(f"Logged in as: {user.display_name}")

        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            online_friends = await friends.get_friends()

            for f in online_friends:
                name = f.display_name
                if name not in FRIEND_NAMES:
                    continue

                if not last_status.get(name, False):
                    location = f.location or "private"
                    world_name = f.world.name if f.world else "Unknown World"
                    instance_id = location if location else "unknown"

                    msg = f"[{now}] **{name}** just logged in!\n" f"World: **{world_name}**\n" f"Instance: `{instance_id}`"
                    print(msg)
                    await send_discord_message(msg)

                last_status[name] = True

            # Mark offline friends
            online_set = {f.display_name for f in online_friends}
            for name in FRIEND_NAMES:
                if name not in online_set:
                    last_status[name] = False

            save_last_status(last_status)
            await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
