from typing import Literal, TypedDict

from pyrogram import types



class purge:
    limit: int
    mode: (Literal["me"] | 
    Literal["companion"] | 
    Literal["both"]
    )


class MessageStorage(TypedDict):
    chat_id: int
    message: types.Message