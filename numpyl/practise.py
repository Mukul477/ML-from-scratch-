import numpy as np
#ndim: number of dimensions
#dtype: data type
#copy: deep copy
#concatenate: join arrays along an axis
#stack: stack arrays along a new axis
#hstack: horizontal stack
#vstack: vertical stack
#split: split array into multiple sub-arrays
#hsplit: horizontal split
#vsplit: vertical split
#where: indices of elements that satisfy a condition
#sort: sort array
import numpy as np

class LinRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, lr=0.01, epochs=100):
        X = np.array(X)
        y = np.array(y)

        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = (X.T @ error) / n
            db = np.mean(error)

            self.w = self.w - lr * dw
            self.b = self.b - lr * db

    def predict(self, X):
        X = np.array(X)
        return X @ self.w + self.b




            

        

        
