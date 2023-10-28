import numpy as np

class LogReg:
    #constructor
    def __init__(self):
        self.theta: np.array | None = None
        self.b = 0

    #used to train the logistic regression model
    def fit(self, x: np.array, y: np.array, epochs=1000, learning_rate=0.0001, verbose=False, print_every_n=10):
        x = x.T
        y = y.reshape((1, -1))
        n = x.shape[0]
        m = x.shape[1]

        self.theta = np.zeros((n, 1))
        for i in range(epochs):
            z = np.dot(self.theta.T, x) + self.b
            y_hat = self.__sigmoid(z)
            assert y_hat.shape == (1, m)

            loss = -(1 / m) * np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
            dW = np.dot(x, (y_hat - y).T) / m
            db = np.mean(y_hat - y)
            self.theta -= learning_rate * dW
            self.b -= learning_rate * db
            if verbose and i % print_every_n == 0:
                print(f'round {i}: loss {loss}')

    def predict_probailities(self, X_test):
        """
        Returns the probabilities P(y = 1 | x_test; theta)
        :param X_test:
        :return:
        """
        X = X_test.T
        Z = np.dot(self.theta.T, X) + self.b
        Y_hat = self.__sigmoid(Z)
        return np.squeeze(Y_hat)

    # predicts the class labels for the input features
    def predict(self, X_test, threshold=0.5):
        yhat = self.predict_probailities(X_test)
        return np.array([1 if p > threshold else 0 for p in yhat])

    @staticmethod
    def accuracy(y_pred, y_true):
        return 1 - np.sum(np.abs(y_pred - y_true)) / len(y_true)

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
        m = [[0, 0], [0, 0]]
        for i in range(len(y)):
            if y[i] == 1:
                if yhat[i] == 1:
                    m[0][0] += 1
                else:
                    m[0][1] += 1
            else:
                if yhat[i] == 1:
                    m[1][0] += 1
                else:
                    m[1][1] += 1
        return m

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

