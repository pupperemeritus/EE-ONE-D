from abc import ABC, abstractmethod


class SearchClass(ABC):
    @classmethod
    @abstractmethod
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def __call__(self):
        pass
