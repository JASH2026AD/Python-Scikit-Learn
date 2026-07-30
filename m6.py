from sklearn.linear_model import LogisticRegression
import numpy as np

# Study hours
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])

# 0 = Fail, 1 = Pass
y = np.array([0,0,0,0,1,1,1,1])

model = LogisticRegression()
model.fit(X, y)

hours = [[5.5]]

prediction = model.predict(hours)
probability = model.predict_proba(hours)

print("Prediction:", prediction[0])
print("Probability:", probability)