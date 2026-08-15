import pandas as pd

# Load dataset
df = pd.read_csv("orders_dataset.csv")

# Shape
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Overall return rate
print("Return rate:", round(df["returned"].mean() * 100, 2), "%")

# Missing ratings %
print("Missing ratings:", round(df["rating_given"].isna().mean() * 100, 2), "%")

# Return rate by product category
print("\nReturn rate by product category:")
print(df.groupby("product_category")["returned"].mean() * 100)

# Return rate by payment method
print("\nReturn rate by payment method:")
print(df.groupby("payment_method")["returned"].mean() * 100)

# MAR justification
cod_missing = df.loc[df["payment_method"] == "COD", "rating_given"].isna().mean() * 100
non_cod_missing = df.loc[df["payment_method"] != "COD", "rating_given"].isna().mean() * 100
print(f"\nMissingness classification: MAR (COD missing {cod_missing:.2f}% vs non-COD {non_cod_missing:.2f}%)")

