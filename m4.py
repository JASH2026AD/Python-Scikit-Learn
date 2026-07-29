from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.DataFrame({
    "Experience": [1, 2, 3, 4, 5, 6],
    "Salary": [30000, 38000, 47000, 55000, 65000, 76000]
})

X = data[["Experience"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X, y)

print("Salary for 7 years:", model.predict([[7]])[0])