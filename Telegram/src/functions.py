from pyrogram import types

from loader import client
from .types import purge


async def purge(message:types.Message, args: purge):
    iterable = await client.get_chat_history(
        chat_id=message.chat.id, 
        limit=args.limit
    )
    
    await client.delete_messages(
        chat_id=message.chat.id,
        message_ids=iterable,
    )