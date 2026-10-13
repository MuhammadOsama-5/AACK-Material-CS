# Lab 6 - Task 3: Weather log wrangling and dashboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Raw, messy 15-day weather log (+1 duplicated row at the end = 16 rows)
raw = {
    "Date": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05",
             "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10",
             "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15",
             "2026-10-05"],
    "City": [" islamabad", "ISLAMABAD ", "Islamabad", "islamabad ", " Lahore",
             "LAHORE", "lahore ", " Islamabad", "ISLAMABAD", "lahore",
             "Lahore ", " LAHORE", "islamabad", "Islamabad ", " lahore",
             " Lahore"],
    "Temperature": ["34C", "26C", "29C", "33C", "30C", "35C", "25C", "28C",
                    "32C", "36C", "29C", "24C", "33C", "30C", "34C", "30C"],
    "Humidity": [45, 82, np.nan, 48, 60, 42, 85, np.nan,
                 50, 40, 63, np.nan, 47, 58, 44, 60],
    "Condition": ["sunny", " Rainy", "CLOUDY ", "Sunny", "cloudy", " SUNNY ",
                  "rainy", "Cloudy", "sunny ", "SUNNY", " cloudy", "Rainy ",
                  "sunny", "CLOUDY", " Sunny", "cloudy"],
}

df = pd.DataFrame(raw)
print("Raw shape:", df.shape)
print(df.head())

# (a) Clean text columns
df["City"] = df["City"].str.strip().str.title()
df["Condition"] = df["Condition"].str.strip().str.title()

# Strip the "C" unit from Temperature and convert to numbers
df["Temperature"] = df["Temperature"].str.strip().str.upper().str.replace("C", "", regex=False)
df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

# Convert Date text to real dates
df["Date"] = pd.to_datetime(df["Date"])

# Drop the duplicate row
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after dropping duplicates:", len(df))

# Fill missing Humidity with the median
print("Missing Humidity before fill:", df["Humidity"].isnull().sum())
median_humidity = df["Humidity"].median()
df["Humidity"] = df["Humidity"].fillna(median_humidity)
print("Missing Humidity filled with median:", median_humidity)

print("\nCleaned data:")
print(df)

# (b) One-hot encode Condition
condition_encoded = pd.get_dummies(df["Condition"], prefix="Condition", dtype=int)
print("\nOne-hot encoded Condition columns:")
print(condition_encoded)
df_encoded = pd.concat([df, condition_encoded], axis=1)

# (d) Correlation between Temperature and Humidity
corr = df["Temperature"].corr(df["Humidity"])
print("\nCorrelation (Temperature, Humidity):", round(corr, 2))

# (e) How many days had each condition
condition_counts = df["Condition"].value_counts()
print("\nCondition counts:")
print(condition_counts)

# (f) All four plots in one 2 x 2 figure
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (c) Line plot: Temperature over Date
axes[0, 0].plot(df["Date"], df["Temperature"], marker="o", color="tomato")
axes[0, 0].set_title("Temperature over Time")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Temperature (°C)")
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# (d) Scatter plot: Temperature vs Humidity
axes[0, 1].scatter(df["Temperature"], df["Humidity"], color="steelblue", s=60)
axes[0, 1].set_title(f"Temperature vs. Humidity (r = {corr:.2f})")
axes[0, 1].set_xlabel("Temperature (°C)")
axes[0, 1].set_ylabel("Humidity (%)")
axes[0, 1].grid(True, alpha=0.3)

# (e) Pie chart: days per condition
axes[1, 0].pie(condition_counts.values, labels=condition_counts.index,
               autopct="%1.1f%%", startangle=90,
               colors=["#F2C14E", "#9AA5B1", "#4C78A8"])
axes[1, 0].set_title("Days per Condition")

# (e) Histogram: Temperature distribution
axes[1, 1].hist(df["Temperature"], bins=5, color="seagreen", edgecolor="white")
axes[1, 1].set_title("Temperature Distribution")
axes[1, 1].set_xlabel("Temperature (°C)")
axes[1, 1].set_ylabel("Number of Days")

fig.suptitle("Weather Overview", fontsize=16)
plt.tight_layout()
plt.savefig("weather_dashboard.png")
print("\nSaved weather_dashboard.png")
plt.show()
