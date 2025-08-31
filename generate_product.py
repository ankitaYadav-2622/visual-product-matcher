import pandas as pd

# Load your full dataset (1100 items)
df = pd.read_csv("product.csv")

# Shuffle rows and select only 180
df_small = df.sample(n=180, random_state=42)

# Save the reduced dataset
df_small.to_csv("products.csv", index=False)

print("✅ Saved 180-row dataset as products_small.csv")
print(df_small.head())
