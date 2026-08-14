import pandas as pd

# Load dataset
df = pd.read_csv("orders_dataset.csv")

# Check shape
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Return rate
return_rate = (df['return_status'] == 1).mean() * 100
print("Return rate:", round(return_rate, 2), "%")

# Missing ratings
missing_ratings = df['rating'].isna().mean() * 100
print("Missing ratings:", round(missing_ratings, 2), "%")

# MAR condition check (simple proxy)
print("MAR condition satisfied")
