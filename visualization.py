import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
print("Loading dataset...")
df = pd.read_csv("books_data_500.csv", encoding='utf-8')

# -------------------------------
# 2. Initial Exploration
# -------------------------------
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# 3. Data Cleaning
# -------------------------------

print("\nCleaning Price column...")

# Extract only numeric value from price (handles Â£ issue)
df["Price"] = df["Price"].str.extract(r'(\d+\.\d+)').astype(float)

# Convert Rating to numeric
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}
df["Rating"] = df["Rating"].map(rating_map)

print("\nCleaned Data:")
print(df.head())

# -------------------------------
# 4. Basic Statistics
# -------------------------------
print("\nStatistical Summary:")
print(df.describe())

print("\nAverage Price:", df["Price"].mean())
print("Maximum Price:", df["Price"].max())
print("Minimum Price:", df["Price"].min())

# -------------------------------
# 5. Price Distribution
# -------------------------------
plt.figure()
sns.histplot(df["Price"], bins=20)
plt.title("Price Distribution of Books")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 6. Rating Distribution
# -------------------------------
plt.figure()
sns.countplot(x="Rating", data=df)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 7. Price vs Rating
# -------------------------------
plt.figure()
sns.boxplot(x="Rating", y="Price", data=df)
plt.title("Price vs Rating")
plt.xlabel("Rating")
plt.ylabel("Price")
plt.show()

# -------------------------------
# 8. Top 10 Expensive Books
# -------------------------------
top_expensive = df.sort_values(by="Price", ascending=False).head(10)

plt.figure()
sns.barplot(x="Price", y="Title", data=top_expensive)
plt.title("Top 10 Most Expensive Books")
plt.xlabel("Price")
plt.ylabel("Book Title")
plt.show()

# -------------------------------
# 9. Cheapest Books
# -------------------------------
cheapest = df.sort_values(by="Price", ascending=True).head(10)

plt.figure()
sns.barplot(x="Price", y="Title", data=cheapest)
plt.title("Top 10 Cheapest Books")
plt.xlabel("Price")
plt.ylabel("Book Title")
plt.show()

# -------------------------------
# 10. Average Price by Rating
# -------------------------------
avg_price_rating = df.groupby("Rating")["Price"].mean()

plt.figure()
avg_price_rating.plot(kind="bar")
plt.title("Average Price by Rating")
plt.xlabel("Rating")
plt.ylabel("Average Price")
plt.show()

# -------------------------------
# 11. Availability Analysis
# -------------------------------
plt.figure()
sns.countplot(x="Availability", data=df)
plt.title("Availability of Books")
plt.xlabel("Stock Status")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 12. Correlation Heatmap
# -------------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 13. Save Cleaned Data
# -------------------------------
df.to_csv("cleaned_books_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# -------------------------------
# 14. Completion Message
# -------------------------------
print("\nData Visualization Completed Successfully!")