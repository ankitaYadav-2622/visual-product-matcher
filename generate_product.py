import os
import csv

images_root = "images"  # your main images folder
output_csv = "product.csv"

products = []
product_id = 1

for category in os.listdir(images_root):
    category_path = os.path.join(images_root, category)
    if not os.path.isdir(category_path):
        continue
    # For each image in the category folder
    for filename in sorted(os.listdir(category_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            product_name = f"{category.replace('_', ' ').title()} {os.path.splitext(filename)[0].split('_')[-1]}"
            imagepath = os.path.join(images_root, category, filename).replace("\\", "/")  # Use forward slash for CSV paths
            price = 100  # Dummy price, you can customize
            products.append([product_id, product_name, category, price, imagepath])
            product_id += 1

# Write to CSV
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["product_id", "product_name", "category", "price", "imagepath"])
    writer.writerows(products)

print(f" '{output_csv}' created with {len(products)} products.")
