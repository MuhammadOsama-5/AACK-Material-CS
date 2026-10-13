# Lab 6 - Task 2: Inventory cleaning, encoding, and plots
import pandas as pd
import matplotlib.pyplot as plt

# Raw, messy data for 10 products
data = {
    "Product": ["Cola", "Chips", "Milk", "Biscuits", "Juice",
                "Yogurt", "Nimko", "Cheese", "Mineral Water", "Butter"],
    "Category": ["beverages ", "SNACKS", " Dairy", "Snacks", "beverages",
                 "DAIRY ", " snacks", "dairy", "BEVERAGES", " Dairy "],
    "Price": ["Rs. 1,250", "Rs. 90", "Rs. 450", "Rs. 120", "Rs. 1,800",
              "Rs. 220", "Rs. 150", "Rs. 1,100", "Rs. 100", "Rs. 650"],
    "Stock": [40, 120, 60, 150, 35, 80, 90, 25, 200, 45],
}

df = pd.DataFrame(data)
print("Raw data:")
print(df)

# (a) Clean the Category text
df["Category"] = df["Category"].str.strip().str.title()

# (b) Convert Price to numbers: remove "Rs. " and the commas, then convert
df["Price"] = (df["Price"]
               .str.replace("Rs. ", "", regex=False)
               .str.replace(",", "", regex=False)
               .astype(float))

# (c) Label encoding of Category (alphabetical: Beverages=0, Dairy=1, Snacks=2)
df["Category_Code"] = df["Category"].astype("category").cat.codes

print("\nCleaned and encoded data:")
print(df)

# (d) Average price per category and share of stock per category
avg_price = df.groupby("Category")["Price"].mean().sort_values(ascending=False)
stock_share = df.groupby("Category")["Stock"].sum()

print("\nAverage price per category:")
print(avg_price.round(2))
print("\nTotal stock per category:")
print(stock_share)

colors = ["#4C78A8", "#F58518", "#54A24B"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(avg_price.index, avg_price.values, color=colors)
axes[0].set_title("Average Price per Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Average Price (Rs.)")
axes[0].grid(axis="y", alpha=0.3)

axes[1].pie(stock_share.values, labels=stock_share.index, autopct="%1.1f%%",
            startangle=90, colors=colors)
axes[1].set_title("Share of Total Stock by Category")

fig.suptitle("Inventory Overview", fontsize=14)
plt.tight_layout()
plt.savefig("inventory_plots.png")
print("\nSaved inventory_plots.png")
plt.show()
