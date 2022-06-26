from pyrogram import Client, types, handlers 


import env
from loader import client, messageStorage
from src.executor import execute_commands


async def message_executor(client: Client, message:types.Message):
    print(message)


async def deleted_message_executor(client: Client, messages: list[types.Message]):
    print(messages)


if __name__ == '__main__':
    client.add_handler(handlers.MessageHandler(message_executor))
    client.add_handler(handlers.DeletedMessagesHandler(deleted_message_executor))
    client.run()