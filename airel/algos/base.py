import abc


class BaseAlgo:
    def __str__(self):
        return self.__class__.__name__
    
    @abc.abstractmethod
    def train(self):
        pass
