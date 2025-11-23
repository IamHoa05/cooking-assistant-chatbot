import json
import pandas as pd
import unicodedata
import re


# ========================================================================
# 1. HÀM XỬ LÝ CHUỖI CƠ BẢN (ICON, KHOẢNG TRẮNG, VIẾT HOA…)
# ========================================================================

def remove_icons(text: str) -> str:
    """Xóa emoji, ký hiệu thuộc Unicode category 'Symbol'."""
    if not text:
        return text

    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\uFE0F"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("S"))
    return re.sub(r"\s+", " ", text).strip()


def clean_json(data):
    """Duyệt toàn bộ JSON và xóa icon."""
    if isinstance(data, dict):
        return {k: clean_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_json(v) for v in data]
    if isinstance(data, str):
        return remove_icons(data)
    return data


def normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi: xóa icon + xóa khoảng trắng thừa."""
    return re.sub(r"\s+", " ", remove_icons(text)).strip()


def normalize_name(text: str) -> str:
    """Viết hoa chữ cái đầu của tên món ăn."""
    if not text:
        return text
    return text.strip().capitalize()


# ========================================================================
# 2. CHUẨN HÓA THỜI GIAN & SỐ NGƯỜI ĂN
# ========================================================================

def normalize_cook_time(time_str):
    """Chuyển '1 giờ 20 phút' → 80 (phút)."""
    if not time_str:
        return 0

    time_str = time_str.lower().strip()
    total = 0

    hours = re.findall(r"(\d+)\s*(giờ|h)", time_str)
    minutes = re.findall(r"(\d+)\s*(phút|ph)", time_str)

    if hours:
        total += int(hours[0][0]) * 60
    if minutes:
        total += int(minutes[0][0])

    if total == 0:
        nums = re.findall(r"\d+", time_str)
        if nums:
            total = int(nums[0])

    return total


def normalize_servings(servings_str):
    """Chuẩn hóa khẩu phần: '4–5 người' → '4-5'."""
    if not servings_str:
        return ""

    servings_str = servings_str.lower().strip()
    match = re.findall(r"(\d+)\s*[-–]?\s*(\d+)?", servings_str)

    if match:
        left, right = match[0]
        return f"{left}-{right}" if right else left

    return ""


# ========================================================================
# 3. CHUẨN HÓA ĐƠN VỊ ĐO LƯỜNG
# ========================================================================

def normalize_unit(text):
    """Chuẩn hóa các đơn vị viết tắt: g → gam, M → muỗng…"""
    if isinstance(text, list):
        return [normalize_unit(x) for x in text]

    if not isinstance(text, str):
        return text

    unit_map = {
        'm': 'muỗng',
        'M': 'muỗng',
        'g': 'gam',
        'kg': 'kilogram',
        'tr': 'trái',
        'c': 'củ',
        'qu': 'quả',
        'ml': 'ml',
    }

    for k, v in unit_map.items():
        text = re.sub(rf"(\d[\d\s./]*)\s*{k}\b", rf"\1 {v}", text)

    return text.strip()


# ========================================================================
# 4. CHUẨN HÓA TÊN NGUYÊN LIỆU (LOẠI BRAND, LOẠI MÔ TẢ)
# ========================================================================

def normalize_ingredient_name(name: str) -> str:
    """Xóa mô tả, thương hiệu, gom nhóm từ đồng nghĩa."""
    if not name:
        return ""

    # 1. Xóa icon + lowercase
    raw = remove_icons(name).lower()

    # 2. Xóa ký tự đặc biệt
    raw = re.sub(r'[^\w\s]', '', raw)

    # # 3. Loại bỏ các từ mô tả/thương hiệu/không cần thiết
    # ignore_terms = ["gia vị", "ăn kèm", "trang trí", "dùng kèm"]
    # for term in ignore_terms:
    #     raw = raw.replace(term, "")

    # 4. Loại bỏ động từ, hành động chế biến
    remove_verbs = [
        "gia vị", "ăn kèm", "ăn trưa kèm", "ăn tối kèm", "trang trí", "dùng kèm", "rau nêm",
        "băm", "phi", "cắt", "xay", "luộc", "thái", "nướng",
        "chiên", "hấp", "trụng", "lát", "nhuyễn", "đập dập", "giã","đập giập",
        "để ráo", "tươi", "sợi", "cắt sợi", "hườm", "poarô", "mềm", "tráng mỏng",
        "bóc vỏ", "non", "già", "cọng", "chín", 'bào', 'trái', "nhỏ", 
        "cây", 'tơ mềm', "dăm", "philê", "tách vỏ", "búp", "khô", 
        "làm sạch","giòn", "nạo", "cọng to", "lặt sạch", "mỏng",
        "lột bỏ da", "khô", "cạn", "nori vuông bằng miếng sandwich",
        "ngâm mềm", "bào mỏng", "sơ","làm sẵn", "ngâm nở", "đát nhỏ",
        "lột vỏ", "số 1", "rang", "ta", "ngon", "có dầu", "dún", "các loại",
        "que", "chần", "cắt hạt lựu", "hạt lựu", "hộp", "mài nhỏ", "còn sống",
        "ngâm dầu", "trái", "đèo", "hình thoi", "nhật", "đã ngâm", "xắt", "lạt", 
        "lớn", "ngâm chua", "giả", "dẻo thơm", "không hạt", "nguyên hạt", "góc tư",
        "nguyên liệu", "bỏ da", "loại", "cac loai", "rút xương", "ruột xanh", 
        "tròn làm đế bánh tiêu", "hột", "đặc ruột", "không da", "loại", "sẵn",
        "đặc", "nguyên vỏ", "da", "thông thường", "nguyên con", "hạt tròn", "vừa tới",
        "ajixốt", "đông lạnh", "đa dụng", "đà", "tùy ý khúc", "tùy ý", "thường", "Ajingon",
        "khúc giữa", "to", "bé", "bỏ vỏ tách đôi", "lon", "để nguyên lá", "mọng", "khoảng",
        "lọc xương", "bỏ vỏ", "gọt vỏ", "khúc", "chừa đuôi", "ngâm nước lạnh", "để riêng gốc và lá",
        "dẹp", "bỏ đuôi", "gọt sạch vỏ", "thả vườn", "ngâm", "áp chảo", "chừa đuôi", "hạt còn vỏ", 
        ""
    ]
    for v in remove_verbs:
        raw = re.sub(rf"\b{v}\b", "", raw)

    # 5. Loại bỏ thương hiệu
    brand_map = ["aji-ngon", "aji-no-moto", "phú sĩ", "ajinomoto"]
    for b in brand_map:
        raw = raw.replace(b, "")

    # 6. Gom nhóm bằng replacements
    replacements = {
        "hạt nêm ajingon heo": "Hạt nêm",
        "hạt nêm ajingon nấm": "Hạt nêm",
        "hạt nêm ajingon gà": "Hạt nêm",
        "bột ngọt ajinomoto": "Bột ngọt",
        "ajinomoto giấm gao len men": "Giấm gạo lên men",
        "nước tương phú sĩ": "Nước tương",
        "nước tương lisa" : "Nước tương",
        "xốt tương đậu nành lisa": "Xốt tương đậu nành",
        "xốt mayonnaise ajimayo vị ngọt dịu": "Xốt Mayonnaise",
        "xốt mayonnaise ajimayo vị nguyên bản": "Xốt Mayonnaise",
        "ajiquick bột": "Bột chiên giòn",
        "ajiquick bột tẩm": "Bột chiên giòn",
        "ajiquick bột tẩm khô giòn": "Bột chiên giòn",
        "ajiquick bột giòn": "Bột chiên giòn",
        "nêm ajiquick lẩu" : "Gia vị nêm sẵn lẩu",
        "nêm sẵn ajiquick lẩu" : "Gia vị nêm sẵn lẩu",
        "nêm sẵn ajiquick thịt kho" :"Gói gia vị nêm sẵn nấu thịt kho",
        "đầu hành và hành tím" : 'Hành', 
        "xốt dùng ngay kho quẹt" : "Kho quẹt",
        "nêm sẵn ajiquick phở bò" : "Gia vị nêm sẵn phở bò",
        "nêm sẵn ajiquick bún riêu cua" : "Gia vị nêm sẵn bún riêu cua",
      

    }
    for k, v in replacements.items():
        # xóa space thừa + lowercase trước khi so sánh
        raw_cmp = re.sub(r'\s+', ' ', raw)
        if k in raw_cmp:
            return v

    # 7. Cleanup khoảng trắng
    raw = re.sub(r'\s+', ' ', raw).strip()
    if not raw:
        return ""

    # 8. Viết hoa chữ cái đầu
    # return raw[0].upper() + raw[1:]
    return raw.lower()


# ========================================================================
# 5. TÁCH NGUYÊN LIỆU → (tên, số lượng)
# ========================================================================

def clean_name(name: str) -> str:
    """Chuẩn hóa tên nguyên liệu cuối cùng."""
    if not name:
        return ""

    name = re.sub(r"\(.*?\)", "", name)
    name = normalize_ingredient_name(name)
    return re.sub(r"\s+", " ", name).strip()


def detect_ingredient_parts(text: str):
    """Tách 1 dòng nguyên liệu → (name, qty) chuẩn hóa nâng cao."""
    text = text.strip()

    # --- 1. Tách nếu có nhiều nguyên liệu bằng dấu phẩy (chỉ lấy phần đầu vì vòng for xử lý từng item) ---
    if "," in text:
        text = text.split(",")[0].strip()

    # --- 2. Nếu có dấu ":" tách name : quantity ---
    if ":" in text:
        name_part, qty_part = text.split(":", 1)
        name = clean_name(name_part)
        qty = qty_part.strip() or None

        # 🔥 CHUẨN HÓA ĐƠN VỊ  (thêm dòng này)
        if qty:
            qty = normalize_unit(qty)

        return name, qty

    # --- 3. Regex tìm số lượng ---
    match = re.search(r"(\d[\d\s./]*\s*(?:g|gam|kg|ml|trái|cây|muỗng|quả|lá)?)", text, flags=re.I)

    if match:
        quantity = match.group(0).strip() or None
        name = text[:match.start()].strip()
        name = clean_name(name)

        # 🔥 CHUẨN HÓA ĐƠN VỊ (thêm dòng này)
        if quantity:
            quantity = normalize_unit(quantity)

        return name, quantity

    # --- 4. Không tìm thấy số lượng → quantity = None ---
    name = clean_name(text)
    return name, None


def process_ingredients(ingredients):
    """Chuyển list nguyên liệu → (list tên, list số lượng)."""
    names, quantities = [], []
    seen = set()  # dùng để loại bỏ trùng lặp

    for item in ingredients:
        name, qty = detect_ingredient_parts(item)
        if not name:
            continue  # skip empty
        if name not in seen:
            names.append(name)
            quantities.append(qty)
            seen.add(name)
        else:
            # nếu muốn gộp qty trùng, xử lý ở đây
            pass

    return names, quantities


# ========================================================================
# 6. PHÂN LOẠI MÓN ĂN
# ========================================================================

def detect_category(name):
    name = name.lower()
    mapping = {
        "canh": "canh", "súp": "súp",
        "xào": "xào", "chiên": "chiên", "rán": "chiên",
        "kho": "kho", "rim": "rim", "om": "om",
        "nướng": "nướng", "hấp": "hấp", "luộc": "luộc",
        "lẩu": "lẩu", "cháo": "cháo",
        "gỏi": "gỏi", "salad": "salad",
        "cuốn": "cuốn", "nem": "nem", "chả": "chả",
        "bún": "món nước", "phở": "món nước",
        "miến": "món nước", "hủ tiếu": "món nước",
        "chè": "chè", "kem": "tráng miệng",
        "bánh": "bánh",
        "cà ri": "cà ri",
        "kim chi": "món Hàn", "tokbokki": "món Hàn",
        "sushi": "món Nhật", "udon": "món Nhật", "ramen": "món Nhật",
        "trộn": "trộn",
        "xốt": "xốt"
    }
    for k, v in mapping.items():
        if k in name:
            return v.capitalize()
    return "món khác"


# ========================================================================
# 7. HÀM CHÍNH XỬ LÝ TOÀN BỘ DATAFRAME
# ========================================================================

def process_and_export(raw_data, output_file):
    df = pd.DataFrame(raw_data)
    
    df["dish_name"] = df["dish_name"].apply(normalize_text).apply(normalize_name)

    df["ingredient_names"], df["ingredient_quantities"] = zip(
        *df["ingredients"].apply(process_ingredients)
    )

    df = df.drop(columns=["ingredients"])

    if "cooking_time" in df.columns:
        df["cooking_time"] = df["cooking_time"].apply(normalize_cook_time)

    if "servings" in df.columns:
        df["servings"] = df["servings"].apply(normalize_servings)

    df["category"] = df["dish_name"].apply(detect_category)

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])

    df = df.reset_index(drop=True)
    df["index"] = df.index + 1
    df["ingredient_count"] = df["ingredient_names"].apply(len)

    df.to_json(output_file, orient="records", indent=2, force_ascii=False)
    print("✅ Đã xuất file", output_file)


# ========================================================================
# 8. CHẠY TRỰC TIẾP
# ========================================================================

if __name__ == "__main__":
    input_file = "./recipes_501_1000_raw.json"
    output_file = "./recipes_501_1000_cleaned.json"

    with open(input_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = clean_json(raw)
    process_and_export(cleaned, output_file)

    # Xóa escape \/ trong URL
    with open(output_file, "r", encoding="utf-8") as f:
        data = f.read().replace("\\/", "/")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(data)

    print("🎉 Hoàn tất.")
