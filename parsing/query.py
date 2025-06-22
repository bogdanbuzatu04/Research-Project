from abc import ABC, abstractmethod

class Query(ABC):


    '''
    This method returns the string if the query matches, else it returns None
    '''
    @abstractmethod
    def contains(self, text, rule):
        pass

    '''
    This method returns the a list of all matches for a query in a string.
    The format is returned as a tuple for each entry consisting of [prefix, match, suffix]
    '''
    @abstractmethod
    def parse(self, text, rule):
        pass