import json
import json5   # pip install json5
import re

def clean_value(value):
    """Làm sạch giá trị"""
    if value in [None, "", [], {}]:
        return None 
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value.strip()
    return value

def normalize_dish_name(name: str) -> str:
    """Chuẩn hóa tên món ăn: xóa 'Phú Sĩ' và viết hoa chữ cái đầu"""
    if not isinstance(name, str) or not name.strip():
        return name
    name = name.replace("Phú Sĩ", "").strip()
    return name.capitalize()

def normalize_ingredient_format(ingredient: str) -> str:
    """
    Chuẩn hóa nguyên liệu về dạng 'Tên: số lượng đơn vị'
    """
    if not isinstance(ingredient, str):
        return ingredient

    ingredient = ingredient.strip()

    # Nếu đã có dấu ':' thì giữ nguyên
    if ":" in ingredient:
        return ingredient

    # Regex tách phần chữ (tên nguyên liệu) và phần số lượng
    match = re.match(r"^(.*?)(\d+\s*[a-zA-ZÀ-ỹ()]+.*)$", ingredient)
    if match:
        name = match.group(1).strip()
        qty  = match.group(2).strip()
        return f"{name}: {qty}"
    
    return ingredient

def clean_dict(d: dict) -> dict: 
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = clean_dict(v)
        elif isinstance(v, list):
            v = clean_list(v, key=k)   # truyền key để xử lý riêng ingredients
        else:
            v = clean_value(v)

        if k == "dish_name" and isinstance(v, str):
            v = normalize_dish_name(v)
        
        if v not in [None, "", [], {}]:
            new_dict[k] = v
    return new_dict

def clean_list(lst: list, key: str = "") -> list:
    """Làm sạch list và loại bỏ trùng lặp"""
    cleaned = []
    seen = set()

    # Regex nhận diện số lượng: ví dụ 50g, 5 tai, 2 quả...
    quantity_pattern = re.compile(r"\d+\s*(g|kg|ml|l|tai|quả|trái|củ|cái|nhánh|chén|muỗng|tép)?", re.IGNORECASE)
    # Danh sách đơn vị hợp lệ cho ingredients
    allowed_units = ["trái", "củ", "quả", "g", "lít", "l"]

    for item in lst:
        if isinstance(item, dict):
            item = clean_dict(item)
        elif isinstance(item, list):
            item = clean_list(item, key=key)
        else:
            item = clean_value(item)

        if key == "ingredients" and isinstance(item, str):
            # Bỏ dòng có 'AJINOMOTO' hoặc 'Phú Sĩ'
            if "AJINOMOTO" in item.upper() or "Phú Sĩ" in item:
                continue
            # Loại bỏ phần trong dấu ()
            item = re.sub(r"\(.*?\)", "", item).strip()
            # Chỉ giữ nguyên liệu có số lượng
            if not quantity_pattern.search(item):
                continue
            # Chỉ giữ nguyên liệu có đơn vị hợp lệ
            if not any(unit in item for unit in allowed_units):
                continue
            # Chuẩn hóa định dạng 'Tên: số lượng đơn vị'
            item = normalize_ingredient_format(item)

        if item not in [None, "", [], {}] and str(item) not in seen:
            cleaned.append(item)
            seen.add(str(item))
    return cleaned


def clean_json_file(input_file: str, output_file: str):
    """
    Đọc JSON, fix lỗi cú pháp (dấu , hoặc { } nếu có), 
    sau đó làm sạch dữ liệu và ghi ra file mới.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = f.read()

    # Dùng json5 để parse kể cả khi có dấu phẩy thừa/thiếu
    try:
        data = json5.loads(raw_data)
    except Exception as e:
        raise ValueError(f"Lỗi parse JSON5: {e}")

    # Làm sạch dữ liệu
    if isinstance(data, dict):
        cleaned = clean_dict(data)
    elif isinstance(data, list):
        cleaned = clean_list(data)
    else:
        raise ValueError("File JSON không phải dạng dict hoặc list")

    # Ghi file mới với JSON chuẩn
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4) 


# --- Chạy thử ---
clean_json_file(
    r"C:\Users\admin\Desktop\recipes_merged.json",
    r"C:\Users\admin\Desktop\CL1.json"
)
