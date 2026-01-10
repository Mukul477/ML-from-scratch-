from logiistic_regression.sigmoid  import LogisticRegression
X = [1,2,3,4,5]
y = [0,0,0,1,1]

model = LogisticRegression()

model.fit(X,y,lr = 0.01,epochs = 1000)

print(model.predict_proba(X))






