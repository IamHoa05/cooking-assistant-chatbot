import json
import re

# Đường dẫn file JSON
input_file = r"C:\Users\admin\Desktop\recipes_merged.json"
output_file = r"C:\Users\admin\Desktop\recipes_filtered.json"

# Regex: chỉ giữ dòng kết thúc bằng trái, g, nhánh, củ, lá, M hoặc m
pattern = re.compile(r"(trái|g|nhánh|củ|lá|M|m)$")

def clean_ingredients(ingredients):
    filtered = []
    for item in ingredients:
        # Thay "1/2" thành "1"
        item = item.replace("1/2", "1").strip()
        # Giữ lại nếu khớp regex
        if pattern.search(item):
            filtered.append(item)
    return filtered

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Nếu file là 1 object
if isinstance(data, dict) and "ingredients" in data:
    data["ingredients"] = clean_ingredients(data["ingredients"])

# Nếu file là list các object
elif isinstance(data, list):
    for dish in data:
        if isinstance(dish, dict) and "ingredients" in dish:
            dish["ingredients"] = clean_ingredients(dish["ingredients"])

# Ghi file JSON đã lọc
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ Đã lọc xong. File kết quả lưu tại:", output_file)
