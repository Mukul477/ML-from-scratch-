from linear_regression.model import LinearRegression

X = [1,2,3,4]
y = [3,6,9,12]

model = LinearRegression()
model.fit(X, y, lr=1, epochs=1000)

print("Predictions:", model.predict(X))
print("Weights:", model.w, "Bias:", model.b)