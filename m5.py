from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40],
    "BMI": [22, 24, 28, 30, 32],
    "Children": [0, 1, 1, 2, 3],
    "Cost": [15000, 18000, 25000, 32000, 42000]
})

X = data[["Age", "BMI", "Children"]]
y = data["Cost"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[29, 26, 2]])
print("Predicted Insurance Cost:", prediction[0])