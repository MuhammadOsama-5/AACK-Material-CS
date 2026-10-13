# Lab 6 - Task 1: Student marks cleaning and bar plot
import pandas as pd
import matplotlib.pyplot as plt

# Raw, messy data (as given in the lab manual)
names = [" ali", "SARA ", "bilal", " Ayesha", "HAMZA", "zain ", "Noor", " usman "]
marks = ["72", "85", "absent", "64", "90", "78", "81", "69"]

df = pd.DataFrame({"Name": names, "Marks": marks})
print("Raw data:")
print(df)

# (a) Clean the Name column: remove extra spaces, fix capitalization
df["Name"] = df["Name"].str.strip().str.title()

# (b) Convert Marks to numbers; "absent" cannot be converted, so it becomes NaN
df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")

# mean() ignores NaN, so this is the mean of the other 7 students
mean_marks = df["Marks"].mean()
df["Marks"] = df["Marks"].fillna(round(mean_marks, 2))

# (c) Print the cleaned DataFrame
print("\nCleaned data:")
print(df)
print("\nAbsent student's mark replaced with mean:", round(mean_marks, 2))

# (d) Bar plot of marks per student
plt.figure(figsize=(8, 5))
plt.bar(df["Name"], df["Marks"], color="steelblue")
plt.title("Marks per Student")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("marks_bar.png")
print("Saved marks_bar.png")
plt.show()
