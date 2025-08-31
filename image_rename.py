import pandas as pd
import os

df = pd.read_csv('product.csv')

def fix_imagepath(row):
    category_folder = row['category'].replace(' ', '_').lower()
    # Extract just filename from current imagepath if exists, else construct filename
    filename = os.path.basename(row['imagepath']) if 'imagepath' in row else f"{category_folder}_{row['product_id']}.jpg"
    # Build new imagepath
    new_path = os.path.join('images', category_folder, filename)
    return new_path.replace('\\', '/')  # For Windows path correction

df['imagepath'] = df.apply(fix_imagepath, axis=1)

df.to_csv('product_updated.csv', index=False)
print(" product_updated.csv saved with corrected image paths")
