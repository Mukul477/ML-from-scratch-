class LinearRegression:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y, lr=0.01, epochs=100):
        pass
    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self.w * x + self.b)
        return preds