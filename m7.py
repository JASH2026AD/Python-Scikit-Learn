from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([
    [0,0],
    [1,0],
    [2,1],
    [3,2],
    [4,3],
    [5,4]
])

y = np.array([0,0,0,1,1,1])

model = LogisticRegression()
model.fit(X,y)

new_email = [[3,3]]

print("Spam Prediction:", model.predict(new_email)[0])
print(model.predict_proba(new_email))