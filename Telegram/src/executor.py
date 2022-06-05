import traceback
from pyrogram import types


import config
from src.commands import commands
from src.parsers import parsers


async def execute_commands(message:types.Message):
    try:
        message_command = message.text.replace(config.command_prefix, '', 1)
        command, args = message_command.split()[0], message_command.split()[1:]
        if message_command in commands:
            func = commands[message_command]
            args = None

        
        elif command in commands:
            func = commands[command]
            args = parsers[func].parse_args(args)

        else:
            return

        await func(message, args)
        
    except Exception as e:
        traceback.print_exc()
        # message = await message.forward(e)
        # await utils.delete_message(message, 5)