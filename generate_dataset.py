import os
import csv
from PIL import Image, ImageDraw, ImageFont
import random

# 25 category names (you can customize)
categories = [
    "tshirts", "sneakers", "watches", "handbags", "jeans",
    "lipsticks", "perfumes", "earrings", "makeup_kits", "nail_polish",
    "phones", "phone_cases", "smartwatches", "tablets", "chargers",
    "hoodies", "jackets", "sandals", "caps", "sunglasses",
    "backpacks", "belts", "skincare", "hair_dryers", "eyeliners"
]

# Directory to save images
base_dir = "images"
os.makedirs(base_dir, exist_ok=True)

# Output CSV
csv_path = "product.csv"
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "name", "category", "price", "imagepath"])

    product_id = 1

    for category in categories:
        category_dir = os.path.join(base_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for i in range(1, 21):  # 20 images per category
            filename = f"{category}_{i}.jpg"
            filepath = os.path.join(category_dir, filename)

            # Create dummy image with text
            img = Image.new('RGB', (256, 256), color=(random.randint(100,255), random.randint(100,255), random.randint(100,255)))
            draw = ImageDraw.Draw(img)
            draw.text((20, 100), f"{category} {i}", fill="black")
            img.save(filepath)

            # Write product to CSV
            name = f"{category.replace('_', ' ').title()} {i}"
            price = random.randint(10, 500)
            writer.writerow([product_id, name, category, price, filepath.replace("\\", "/")])
            product_id += 1

print(f" Dataset generated: {len(categories)} categories × 20 images")
print(f" Images saved in: {base_dir}/")
print(f"  CSV saved as: {csv_path}")
