import math

class LogisticRegression:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y, lr=0.01, epochs=100):
        n = len(X)
        for _ in range(epochs):
            dw = 0.0
            db = 0.0
            for i in range(n):
                z = self.w * X[i] + self.b
                y_hat = self.sigmoid(z)
                error = y_hat - y[i]
                dw += error * X[i]
                db += error
            dw /= n
            db /= n
            self.w -= lr * dw
            self.b -= lr * db

    def predict_proba(self, X):
        probs = []
        for x in X:
            z = self.w * x + self.b
            probs.append(self.sigmoid(z))
        return probs

    def predict(self, X, threshold=0.5):
        preds = []
        for p in self.predict_proba(X):
            preds.append(1 if p >= threshold else 0)
        return preds
            

        
        

