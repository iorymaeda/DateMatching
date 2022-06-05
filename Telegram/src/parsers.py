import argparse

import src.functions as F


class ArgumentParser(argparse.ArgumentParser):    
    def error(self, message):
        raise Exception(message)

    def exit(self, status, message):
        pass


parsers = {}

parser = ArgumentParser(exit_on_error=False)
parser.add_argument('limit', type=int, action='store')
parser.add_argument('-mode', type=str, action='store', default='me')
parsers[F.purge] = parser