from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.DataFrame({
    "Area": [1000, 1200, 1500, 1800, 2000],
    "Bedrooms": [2, 2, 3, 3, 4],
    "Price": [30, 36, 45, 54, 60]
})

X = data[["Area", "Bedrooms"]]
y = data["Price"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[1600, 3]])
print("Predicted Price:", prediction[0], "lakhs")