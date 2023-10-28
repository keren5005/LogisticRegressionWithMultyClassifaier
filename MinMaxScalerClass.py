import numpy as np

"""
Min-max scaling is a common preprocessing technique used to 
scale numeric features to a specific range, typically between 0 and 1.
"""
#initializes the attributes self.__min and self.__max to zero
class MinMaxScaler:
    def __init__(self):
        self.__min = 0
        self.__max = 0
    #fit the scaler on the data and transform it simultaneously
    def fit_transform(self, x):
        self.__min = np.min(x)
        self.__max = np.max(x)
        return (x - self.__min) / (self.__max - self.__min);
