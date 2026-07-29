from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6],
    "Marks": [35, 45, 50, 60, 70, 80]
})

X = data[["Hours"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X, y)

print("Marks for 7 hours:", model.predict([[7]])[0])