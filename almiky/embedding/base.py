'''
Basic embedder
'''
from abc import ABC, abstractmethod


class Embedder(ABC):

    @abstractmethod
    def embed(self, *args, **kwargs):
        pass

    @abstractmethod
    def extract(self, *args, **kwargs):
        pass
