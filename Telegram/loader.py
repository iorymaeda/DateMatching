from pyrogram import Client

import env
from src.types import MessageStorage


messageStorage: MessageStorage = {}
client = Client("UserBot", api_id=env.api_id, api_hash=env.api_hash)
