import numpy as np
from LogisticRegressionClass import LogReg

"""
the class is used for multiclass logistic regression using the one-vs-all approach
"""
class OneVsAll:
    def __init__(self):
        self.__predictors = {}
        self.__all_classes = []

    #This method fits the logistic regression models for each class in the dataset
    def fit(self, X: np.array, Y: np.array, epochs=1000, learning_rate=0.1, verbose=False, print_every_n=10):
        y_vals = np.unique(Y)
        self.__all_classes = y_vals
        n = len(y_vals)
        for i in range(n - 1):
            my_class = y_vals[i]
            yy = []
            for v in Y:
                if v == my_class:
                    yy.append(1)
                else:
                    yy.append(0)
            if verbose:
                print('Predictor ', i)
            regressor = LogReg()
            regressor.fit(X, np.array(yy), epochs=epochs, learning_rate=learning_rate, verbose=verbose,
                          print_every_n=print_every_n)
            self.__predictors[my_class] = regressor

    #This method predicts the class probabilities for input samples x
    def predict_probabilities(self, x):
        out = {}
        for my_class in self.__predictors.keys():
            out[my_class] = self.__predictors[my_class].predict_probailities(x)
        return out

    #This method predicts the class labels for input samples x
    def predict(self, x, threshold=0.5):
        out = self.predict_probabilities(x)
        yhat = None

        probs = out[self.__all_classes[0]]
        if yhat is None:
            yhat = np.zeros(len(probs))
        for i in range(len(yhat)):
            dct = {}
            for k in out.keys():
                dct[k] = out[k][i]
            yhat[i] = self.__get_prob_multi(dct, threshold)
        return yhat

    @staticmethod
    def accuracy(y_pred, y_true):
        return np.count_nonzero(y_pred == y_true) / len(y_true)

    #This method calculates the accuracy for each class separately and returns a dictionary
    # where the keys are the class labels, and the values are the corresponding accuracy scores.
    def accuracy_by_class(self, y_pred, y_true):
        out = {}
        for class_value in self.__all_classes:

            good = 0
            total = 0
            for i in range(len(y_true)):
                if y_true[i] == class_value:
                    total += 1
                    if y_true[i] == y_pred[i]:
                        good += 1
            a = good / total
            out[class_value] = a
        return out

    #This private method is used internally by the
    # predict method to assign the class label based on the predicted probabilities.
    def __get_prob_multi(self, out_dict, threshold):
        for my_class in self.__all_classes:
            if my_class in out_dict.keys():
                if out_dict[my_class] >= threshold:
                    return my_class
            else:
                return my_class

    def confusion_matrix(self, x, y, threshold=0.5):
        #               - ACTUAL -
        # Predicted |  True      | False
        #    -------------------------------
        #    True   |  True Pos  | False Pos
        #           |            |
        #    --------------------------------
        #    False  |  False Neg | True Neg
        #           |            |

        # True positive rate = (True positive) / (True Positive + False Neg)
        # False Positive rate = (False Pos) / (False Pos + True Neg)
        yhat = self.predict(x, threshold)
        big_matrix = []
        for y_true in self.__all_classes:
            m = [[0, 0], [0, 0]]
            for i in range(len(y)):
                if y[i] == y_true:
                    if yhat[i] == 1:
                        m[0][0] += 1
                    else:
                        m[0][1] += 1
                else:
                    if yhat[i] == 1:
                        m[1][0] += 1
                    else:
                        m[1][1] += 1
            big_matrix.append(m)
        return big_matrix

    def print_confusion_matrix(self, M, tr):
        for i in range(len(self.__all_classes)):
            my_class = self.__all_classes[i]
            m = M[i]
            tp = m[0][0]
            fn = m[0][1]
            fp = m[1][0]
            tn = m[1][1]
            print(f"------- Y = {tr.inv_transform(my_class)} ------------")
            print(f"{tp} | {fp}")
            print(f"{fn} | {tn}")
        print("--------------------")
