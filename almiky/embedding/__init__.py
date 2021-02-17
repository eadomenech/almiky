'''Embedders'''

from abc import ABC, abstractmethod


class Embedder(ABC):

    @abstractmethod
    def embed(amplitude, bit):
        pass

    @abstractmethod
    def extract(amplitude):
        pass
