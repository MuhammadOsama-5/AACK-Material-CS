# Task 1 - Estimate Housing Prices Near Transit
# Dataset: UCI Real Estate Valuation (414 transactions)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]
print("Rows used:", len(homes))

# (a) Fit linear regression on all 414 rows
model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope:.4f} price units per 100 m")
print(f"Intercept: {intercept:.4f} price units")

# (b) Predict price per unit area at 500 m (5 x 100 m)
new_home = pd.DataFrame({"distance_100m": [5]})
prediction = model.predict(new_home)[0]
print(f"Predicted price at 500 m: {prediction:.2f} price units")

# (c) Explain the slope
print(f"Each extra 100 m from the MRT station lowers the predicted price by "
      f"about {abs(slope):.2f} units (about {abs(slope) * 10000:,.0f} NT$ per ping).")

# Plot observations and the fitted line
line_x = pd.DataFrame({"distance_100m": [X["distance_100m"].min(), X["distance_100m"].max()]})
line_y = model.predict(line_x)

plt.figure(figsize=(8, 5))
plt.scatter(X["distance_100m"], y, alpha=0.5, label="Observed transactions")
plt.plot(line_x["distance_100m"], line_y, color="red", linewidth=2, label="Fitted line")
plt.xlabel("Distance to nearest MRT station (100 m)")
plt.ylabel("Price per unit area (10,000 NT$ per ping)")
plt.title("House price vs distance to MRT station")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png", dpi=150)
print("Saved task1_housing.png")
