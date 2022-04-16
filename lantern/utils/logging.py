from colorama import Fore, Style

from abc import ABC, abstractmethod


class Logger(ABC):

    @abstractmethod
    def warn(self, *args):
        pass

    @abstractmethod
    def info(self, *args):
        pass

    @abstractmethod
    def success(self, *args):
        pass

    @abstractmethod
    def error(self, *args):
        pass


class SimpleLogger(Logger):
    def __init__(self):
        super().__init__()

    def warn(self, *args):
        print(Fore.YELLOW, *args)
        print(Style.RESET_ALL)

    def info(self, *args):
        print(*args)
        print(Style.RESET_ALL)

    def success(self, *args):
        print(Fore.CYAN, *args)
        print(Style.RESET_ALL)

    def error(self, *args):
        print(Fore.RED, *args)
        print(Style.RESET_ALL)
