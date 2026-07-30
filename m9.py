from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([
    [20000],
    [25000],
    [30000],
    [35000],
    [45000],
    [55000],
    [70000]
])

y = np.array([0,0,0,1,1,1,1])

model = LogisticRegression()
model.fit(X,y)

income = [[40000]]

print("Buy Product:", model.predict(income)[0])
print(model.predict_proba(income))