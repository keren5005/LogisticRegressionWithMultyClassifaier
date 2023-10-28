import numpy as np
"""
the OrdinalEncoder class is used to encode categorical variables into ordinal integers.
"""
class OrdinalEncoder:
    #class constructor that initializes the __dictionary attribute as an empty dictionary.
    def __init__(self):
        self.__dictionary = {}

    # This method takes an ordinal integer y and returns the
    # corresponding category based on the mapping stored in the __dictionary attribute
    def inv_transform(self, y):
        for k in self.__dictionary.keys():
            if self.__dictionary[k] == y:
                return k
        return None

    #This method takes an array-like object x
    # representing the categorical variables and encodes them into ordinal integers
    def fit_transform(self, x) -> np.array:
        out = []
        for e in x:
            if e not in self.__dictionary.keys():
                key = len(self.__dictionary)
                self.__dictionary[e] = key
                out.append(key)
            else:
                out.append(self.__dictionary[e])
        return np.array(out)
