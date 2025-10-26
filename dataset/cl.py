import json5
import re

# Danh sách từ khóa cần xóa
remove_keywords = ["Phú Sĩ", "AJI-NO-MOTO", "Aji-ngon® Heo", "AJINOMOTO", "gia vị", "ĂN KÈM"]

# Danh sách đơn vị hợp lệ
valid_units = ["g", "củ", "trái", "ml", "cây", "bắp", "nhánh", "M", "m"]

def clean_string(s: str) -> str:
    """Xóa ký tự không hợp lệ, nội dung trong ngoặc và khoảng trắng thừa"""
    # Xóa cả nội dung trong dấu ngoặc tròn (....)
    s = re.sub(r"\(.*?\)", "", s)
    # Xóa các dấu . , " ' không cần thiết
    s = re.sub(r"[.,\"']", "", s)
    # Xóa khoảng trắng thừa
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_value(value):
    """Chuẩn hóa giá trị"""
    if value in [None, "", [], {}]:
        return None
    if isinstance(value, str):
        value = clean_string(value)
        return value
    elif isinstance(value, (int, float)):
        return float(value)
    return value

def has_valid_unit(text: str) -> bool:
    """Kiểm tra xem chuỗi có chứa đơn vị hợp lệ không"""
    return any(re.search(rf"\b{unit}\b", text) for unit in valid_units)

def filter_ingredients(ingredients):
    """Giữ lại nguyên liệu có đơn vị hợp lệ và không chứa từ khóa cấm"""
    result = []
    for ing in ingredients:
        ing = clean_value(str(ing))
        # Nếu chứa từ khóa cấm thì bỏ
        if any(keyword.lower() in ing.lower() for keyword in remove_keywords):
            continue
        # Nếu có đơn vị hợp lệ thì giữ, ngược lại bỏ
        if has_valid_unit(ing):
            result.append(ing)
    return result

def clean_data(data):
    """Làm sạch toàn bộ dữ liệu JSON"""
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k == "ingredients" and isinstance(v, list):
                new_dict[k] = filter_ingredients(v)
            else:
                new_dict[k] = clean_data(v)
        return new_dict
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    else:
        return clean_value(data)

# --- Chạy thử ---
input_file = r"C:\Users\admin\Desktop\recipes_merged.json"
output_file = r"C:\Users\admin\Desktop\CL.json"

with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json5.load(f)

clean_data_result = clean_data(raw_data)

with open(output_file, "w", encoding="utf-8") as f:
    json5.dump(clean_data_result, f, ensure_ascii=False, indent=2)

print("✅ Đã lọc: chỉ giữ nguyên liệu có đơn vị hợp lệ và lưu vào", output_file)
