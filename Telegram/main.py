from pyrogram import Client, types, handlers 


import env
from loader import client, messageStorage
from src.executor import execute_commands


async def message_executor(client: Client, message:types.Message):
    messageStorage
    user: types.User = message.from_user
    if user.id == env.owner:
        await execute_commands(message)


async def deleted_message_executor(client: Client, messages: list[types.Message]):
    print(messages)


if __name__ == '__main__':
    client.add_handler(handlers.MessageHandler(message_executor))
    client.add_handler(handlers.DeletedMessagesHandler(deleted_message_executor))
    client.run()