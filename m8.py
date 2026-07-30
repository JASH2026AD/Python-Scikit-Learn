from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[80],[90],[100],[110],[130],[150],[180]])

y = np.array([0,0,0,0,1,1,1])

model = LogisticRegression()
model.fit(X,y)

patient = [[140]]

print("Diabetes:", model.predict(patient)[0])
print(model.predict_proba(patient))