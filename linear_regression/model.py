
class LinearRegression:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y, lr=0.01, epochs=100):
        n = len(X)
        for _ in range(epochs):
            dw = 0.0
            db = 0.0
            for i in range(n):
                y_pred = self.w * X[i] + self.b
                error = y_pred - y[i]

                dw += error * X[i]
                db += error
            dw /= n 
            db /= n
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self.w * x + self.b)
        return preds