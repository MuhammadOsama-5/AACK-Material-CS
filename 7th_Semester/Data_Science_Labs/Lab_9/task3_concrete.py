# Task 3 - Tune a Concrete Strength Model
# Dataset: UCI Concrete Compressive Strength

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]
print("Rows after removing duplicates:", len(concrete))

# (a) Hold out 20% test set, then CV the linear baseline on training rows
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

baseline_rmse = -cross_val_score(LinearRegression(), X_train, y_train, cv=folds, scoring=scoring)
print("Linear baseline fold RMSEs:", baseline_rmse.round(2))
print(f"Linear baseline mean CV RMSE: {baseline_rmse.mean():.2f} MPa")

# (b) Tune polynomial + scaling + Ridge on the same folds
ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])
grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}
search = GridSearchCV(ridge_model, grid, cv=folds, scoring=scoring, refit=True)
search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_)
top3 = results.sort_values("rank_test_score").head(3)
print("\nTop 3 settings:")
for _, row in top3.iterrows():
    print(f"  Rank {row['rank_test_score']}: degree={row['param_poly__degree']}, "
          f"alpha={row['param_ridge__alpha']}, mean CV RMSE={-row['mean_test_score']:.2f} MPa")

print("Best parameters:", search.best_params_)
print(f"Best mean CV RMSE: {-search.best_score_:.2f} MPa")

# (c) Test the refitted winner on the held-out rows
test_predictions = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mae = mean_absolute_error(y_test, test_predictions)
print(f"\nTest RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE: {test_mae:.2f} MPa")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_predictions, alpha=0.6, label="Test rows")
low = min(y_test.min(), test_predictions.min())
high = max(y_test.max(), test_predictions.max())
plt.plot([low, high], [low, high], color="red", linestyle="--", label="y = x")
plt.xlabel("Actual strength (MPa)")
plt.ylabel("Predicted strength (MPa)")
plt.title("Tuned Ridge model: actual vs predicted")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=150)
print("Saved task3_concrete.png")

# (d) Comparison and one interaction term
improvement = baseline_rmse.mean() - (-search.best_score_)
print(f"\nTuned model lowers mean CV RMSE by {improvement:.2f} MPa compared with the baseline.")
poly_names = PolynomialFeatures(degree=2, include_bias=False).fit(X_train).get_feature_names_out()
print("Example degree-2 interaction term:", poly_names[list(poly_names).index("Cement Water")])
