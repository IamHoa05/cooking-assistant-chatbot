import json
import unicodedata
import re
import os
import pandas as pd
from typing import Dict, List, Tuple, Any, Union

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


def clean_json(data: Any) -> Any:
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


INGREDIENT_CORRECTIONS = {
    # Sửa lỗi chính tả phổ biến - THỊT & HẢI SẢN
    "bo": "bò", "heo": "thịt heo", "ga": "gà", "vit": "vịt", "lon": "thịt heo",
    "ca": "cá", "tom": "tôm", "muc": "mực", "cua": "cua", "ech": "ếch",
    "thit": "thịt", "tht": "thịt", "tht bo": "thịt bò", "tht heo": "thịt heo",
    "tht ga": "thịt gà", "tht vit": "thịt vịt", "tht ech": "thịt ếch",
    "ba chi": "ba chỉ", "ba roi": "ba rọi", "ba rọi": "ba rọi", "ba roi": "ba rọi",
    "ba chi bo": "ba chỉ bò", "ba chi heo": "ba chỉ heo",
    "bap bo": "bắp bò", "bap bo hoa": "bắp bò hoa",
    "than bo": "thăn bò", "than heo": "thăn heo",
    "dui ga": "đùi gà", "uc ga": "ức gà", "uc vit": "ức vịt",
    "canh ga": "cánh gà", "dui ech": "đùi ếch",
    "bo bam": "bò bằm", "heo bam": "heo bằm", "ga bam": "gà bằm",
    "bo vien": "bò viên", "ca vien": "cá viên", "tom vien": "tôm viên",
    "cha": "chả", "cha bo": "chả bò", "cha ca": "chả cá", "cha tom": "chả tôm",
    "cha lua": "chả lụa", "cha gio": "chả giò", "cha que": "chả quế",
    
    # HẢI SẢN & THỦY SẢN
    "ca basa": "cá basa", "ca hoi": "cá hồi", "ca thu": "cá thu", "ca loc": "cá lóc",
    "ca dieu hong": "cá điêu hồng", "ca chem": "cá chẽm", "ca bong lau": "cá bông lau",
    "ca bop": "cá bớp", "ca rot": "cá rô", "ca tre": "cá trê", "ca chinh": "cá chình",
    "ca com": "cá cơm", "ca nac": "cá nục", "ca sac": "cá sặc", "ca keo": "cá kèo",
    "ca duoi": "cá đuối", "ca doi": "cá đối", "ca minh": "cá mình",
    "tom su": "tôm sú", "tom the": "tôm thẻ", "tom bac": "tôm bạc", "tom dat": "tôm đất",
    "tom cang": "tôm càng", "tom cang xanh": "tôm càng xanh",
    "muc ong": "mực ống", "muc la": "mực lá", "muc nu": "mực nang", "muc sua": "mực sữa",
    "muc trung": "mực trứng", "muc xao": "mực xào",
    "cua bien": "cua biển", "cua gach": "cua gạch", "cua lot": "cua lột",
    "ghe": "ghẹ", "ghe song": "ghẹ sống",
    "so": "sò", "so huyet": "sò huyết", "so long": "sò lông", "so diep": "sò điệp",
    "ngheu": "nghêu", "ngheu sua": "nghêu sữa", "ngheu song": "nghêu sống",
    "hau": "hàu", "hau sua": "hàu sữa", "hau song": "hàu sống",
    "oc": "ốc", "oc buou": "ốc bươu", "oc huong": "ốc hương", "oc mong tay": "ốc móng tay",
    "oc gac": "ốc giác", "oc lac": "ốc lác", "oc mo": "ốc mỡ",
    
    # RAU CỦ & TRÁI CÂY
    "rau": "rau", "cu": "củ", "la": "lá", "qua": "quả", "trai": "trái",
    "rau muong": "rau muống", "rau mong toi": "rau mồng tơi", "rau den": "rau dền",
    "rau day": "rau đay", "rau lang": "rau lang", "rau ngo": "rau ngổ",
    "rau om": "rau om", "rau ram": "rau răm", "rau ma": "rau má",
    "rau can": "rau cần", "rau can tau": "rau cần tàu", "rau can tay": "rau cần tây",
    "rau thom": "rau thơm", "rau hung": "rau húng", "rau hung que": "rau húng quế",
    "rau hung lui": "rau húng lủi", "rau hung chan": "rau húng chanh",
    "rau tia to": "rau tía tô", "rau ngo gai": "rau ngò gai", "rau ngo ri": "rau ngò rí",
    "rau ngo om": "rau ngò om", "rau kinh gioi": "rau kinh giới",
    "rau xa lach": "rau xà lách", "xa lach": "xà lách", "sa lach": "xà lách",
    "rau cai": "rau cải", "cai thao": "cải thảo", "cai trang": "cải trắng",
    "cai xanh": "cải xanh", "cai ngot": "cải ngọt", "cai be xanh": "cải bẹ xanh",
    "cai beo": "cải bó xôi", "cai bo xoi": "cải bó xôi", "cai xoong": "cải xoong",
    "cai kale": "cải kale", "cai lan": "cải làn", "cai rong": "cải rổ",
    "bap cai": "bắp cải", "bap cai trang": "bắp cải trắng", "bap cai tim": "bắp cải tím",
    "bap cai xanh": "bắp cải xanh", "bap cai baby": "bắp cải baby",
    "su hao": "su hào", "su su": "su su", "bi dao": "bí đao", "bi do": "bí đỏ",
    "bi ngoi": "bí ngòi", "bi ngoi xanh": "bí ngòi xanh", "bi ngoi vang": "bí ngòi vàng",
    "kho qua": "khổ qua", "kho qua rung": "khổ qua rừng", "kho qua tay": "khổ qua tây",
    "muop": "mướp", "muop huong": "mướp hương", "muop kia": "mướp khía",
    "bau": "bầu", "bau sao": "bầu sao", "dau bap": "đậu bắp", "dau ve": "đậu ve",
    "dau cove": "đậu côve", "dau que": "đậu que", "dau rong": "đậu rồng",
    "dau dua": "đậu đũa", "dau bi": "đậu bi", "dau ha lan": "đậu hà lan",
    "ca chua": "cà chua", "ca chua bi": "cà chua bi", "ca chua do": "cà chua đỏ",
    "ca chua vang": "cà chua vàng", "ca phao": "cà pháo", "ca tim": "cà tím",
    "ca rot": "cà rốt", "ca rot baby": "cà rốt baby", "ca rot bi": "cà rốt bi",
    "cu cai": "củ cải", "cu cai trang": "củ cải trắng", "cu cai do": "củ cải đỏ",
    "cu cai duong": "củ cải đường", "cu cai muoi": "củ cải muối",
    "cu den": "củ dền", "cu sen": "củ sen", "cu san": "củ sắn", "cu tu": "củ từ",
    "cu gung": "củ gừng", "cu nghe": "củ nghệ", "cu rieng": "củ riềng",
    "cu hanh": "củ hành", "hanh tay": "hành tây", "hanh tim": "hành tím",
    "hanh la": "hành lá", "hanh khu": "hành khô", "hanh phi": "hành phi",
    "hanh boaro": "hành boa rô", "hanh poaro": "hành boa rô", "hanh baro": "hành boa rô",
    "toi": "tỏi", "toi bac": "tỏi bắc", "toi ly son": "tỏi lý sơn",
    "ot": "ớt", "ot hiem": "ớt hiểm", "ot sung": "ớt sừng", "ot chuo": "ớt chuông",
    "ot chuo do": "ớt chuông đỏ", "ot chuo vang": "ớt chuông vàng", "ot chuo xanh": "ớt chuông xanh",
    "sa": "sả", "sa cay": "sả cây", "sa bam": "sả bằm",
    "gung": "gừng", "nghe": "nghệ", "rieng": "riềng", "xa": "sả",
    "la chanh": "lá chanh", "la que": "lá quế", "la dua": "lá dứa",
    "la lot": "lá lốt", "la giang": "lá giang", "la cam": "lá cẩm",
    "la non": "lá nón", "la ech": "lá é", "la tia to": "lá tía tô",
    
    # TRÁI CÂY
    "chuoi": "chuối", "chuoi xanh": "chuối xanh", "chuoi chin": "chuối chín",
    "chuoi cau": "chuối cau", "chuoi chat": "chuối chát", "chuoi huong": "chuối hương",
    "chuoi sua": "chuối sứ", "chuoi sap": "chuối sáp", "chuoi laba": "chuối laba",
    "cam": "cam", "cam mat": "cam mật", "cam vang": "cam vàng", "cam my": "cam mỹ",
    "quyt": "quýt", "quyt ngot": "quýt ngọt",
    "chanh": "chanh", "chanh vang": "chanh vàng", "chanh xanh": "chanh xanh",
    "chanh bac": "chanh bắc", "chanh muoi": "chanh muối", "chanh day": "chanh dây",
    "buoi": "bưởi", "buoi nam roi": "bưởi năm roi", "buoi xanh": "bưởi xanh",
    "xoai": "xoài", "xoai cat": "xoài cát", "xoai tuong": "xoài tượng",
    "xoai xanh": "xoài xanh", "xoai chin": "xoài chín",
    "man": "mận", "man do": "mận đỏ", "man den": "mận đen",
    "dao": "đào", "dao tien": "đào tiên", "dao my": "đào mỹ",
    "mo": "mơ", "mo dong": "mơ đóng",
    "mit": "mít", "mit tot": "mít tốt", "mit dai": "mít dai",
    "mang cut": "măng cụt", "mang cau": "mãng cầu",
    "nho": "nho", "nho xanh": "nho xanh", "nho do": "nho đỏ", "nho den": "nho đen",
    "nho mong tay": "nho móng tay", "nho my": "nho mỹ",
    "tao": "táo", "tao do": "táo đỏ", "tao xanh": "táo xanh", "tao tau": "táo tàu",
    "le": "lê", "le han quoc": "lê hàn quốc", "le vang": "lê vàng",
    "dua": "dừa", "dua xiem": "dừa xiêm", "dua ram": "dừa rám",
    "dua hau": "dưa hấu", "dua le": "dưa lê", "dua luoi": "dưa lưới",
    "dua leo": "dưa leo", "dua leo baby": "dưa leo baby",
    "dua chuo": "dưa chuột", "dua gang": "dưa gang",
    "dua hau": "dưa hấu", "dua hau do": "dưa hấu đỏ", "dua hau vang": "dưa hấu vàng",
    "thanh long": "thanh long", "thanh long do": "thanh long đỏ", "thanh long trang": "thanh long trắng",
    
    # GIA VỊ & NƯỚC CHẤM
    "gia vi": "gia vị", "gia vi nem": "gia vị nêm",
    "nuoc mam": "nước mắm", "nuoc mam ngon": "nước mắm ngon",
    "nuoc tuong": "nước tương", "nuoc tuong do": "nước tương đổ",
    "tuong": "tương", "tuong ot": "tương ớt", "tuong ca": "tương cà",
    "tuong den": "tương đen", "tuong ximuoi": "tương xí muội",
    "tuong ban": "tương bần", "tuong me": "tương me",
    "mam": "mắm", "mam tom": "mắm tôm", "mam nem": "mắm nêm",
    "mam ruoc": "mắm ruốc", "mam tep": "mắm tép", "mam sac": "mắm sặc",
    "mam ca lin": "mắm cá linh", "mam bo hoc": "mắm bò hóc",
    "muoi": "muối", "muoi hat": "muối hạt", "muoi hong": "muối hồng",
    "muoi ot": "muối ớt", "muoi ot xanh": "muối ớt xanh",
    "duong": "đường", "duong cat": "đường cát", "duong phen": "đường phèn",
    "duong nau": "đường nâu", "duong den": "đường đen",
    "tieu": "tiêu", "tieu den": "tiêu đen", "tieu xanh": "tiêu xanh",
    "tieu do": "tiêu đỏ", "tieu soi": "tiêu sọ", "tieu hat": "tiêu hạt",
    "bot ngo": "bột ngọt", "bot ngo ajinomoto": "bột ngọt",
    "hat nem": "hạt nêm", "hat nem heo": "hạt nêm heo", "hat nem ga": "hạt nêm gà",
    "hat nem nam": "hạt nêm nấm", "hat nem aji": "hạt nêm",
    "dau an": "dầu ăn", "dau me": "dầu mè", "dau oliu": "dầu ô liu",
    "dau dau nanh": "dầu đậu nành", "dau phong": "dầu phộng",
    "dau dieu": "dầu điều", "dau mau": "dầu màu",
    "giam": "giấm", "giam gao": "giấm gạo", "giam tao": "giấm táo",
    "giam trang": "giấm trắng", "giam balsamic": "giấm balsamic",
    "sa te": "sa tế", "sa te tom": "sa tế tôm",
    "tuong ot han quoc": "tương ớt hàn quốc", "tuong ot samyang": "tương ớt samyang",
    "tuong ot chin su": "tương ớt chinsu",
    
    # BỘT & ĐỒ KHÔ
    "bot": "bột", "bot mi": "bột mì", "bot nep": "bột nếp", "bot gao": "bột gạo",
    "bot nang": "bột năng", "bot bot": "bột bắp", "bot chien gion": "bột chiên giòn",
    "bot chien xu": "bột chiên xù", "bot banh bao": "bột bánh bao",
    "bot banh xeo": "bột bánh xèo", "bot banh ran": "bột bánh rán",
    "bot banh cuon": "bột bánh cuốn", "bot banh khot": "bột bánh khọt",
    "bot ngot": "bột ngọt", "bot ca ri": "bột cà ri", "bot quay": "bột quế",
    "bot toi": "bột tỏi", "bot hanh": "bột hành", "bot ot": "bột ớt",
    "bot ot paprika": "bột ớt paprika", "bot ot han quoc": "bột ớt hàn quốc",
    "bot matcha": "bột matcha", "bot cacao": "bột cacao", "bot tra xanh": "bột trà xanh",
    "bot rau cau": "bột rau câu", "bot jelly": "bột jelly",
    "bot baking soda": "bột baking soda", "bot no": "bột nở", "bot noi": "bột nổi",
    "mi chinh": "mì chính",
    
    # SẢN PHẨM TỪ SỮA & TRỨNG
    "trung": "trứng", "trung ga": "trứng gà", "trung vit": "trứng vịt",
    "trung cut": "trứng cút", "trung muoi": "trứng muối",
    "trung bac thao": "trứng bắc thảo", "trung lon": "trứng lộn",
    "sua": "sữa", "sua tuoi": "sữa tươi", "sua dac": "sữa đặc",
    "sua chua": "sữa chua", "sua chua khong duong": "sữa chua không đường",
    "sua dau nanh": "sữa đậu nành", "sua hat": "sữa hạt",
    "pho mai": "phô mai", "pho mai con bo cuoi": "phô mai con bò cười",
    "pho mai mozzarella": "phô mai mozzarella", "pho mai cheddar": "phô mai cheddar",
    "pho mai parmesan": "phô mai parmesan", "pho mai kem": "phô mai kem",
    "kem": "kem", "kem tuoi": "kem tươi", "kem whipping": "kem whipping",
    "kem topping": "kem topping", "kem vani": "kem vani",
    "bơ": "bơ", "bo": "bơ", "bo la": "bơ lạt", "bo man": "bơ mặn",
    "bo thuc vat": "bơ thực vật", "bo margarine": "bơ margarine",
    "bo dau phong": "bơ đậu phộng", "bo me": "bơ mè",
    
    # ĐẬU & HẠT
    "dau hu": "đậu hũ", "dau hu trang": "đậu hũ trắng", "dau hu chien": "đậu hũ chiên",
    "dau hu nuoc": "đậu hũ nước", "dau hu ky": "đậu hũ ky", "dau hu ki": "đậu hũ ky",
    "dau phong": "đậu phộng", "dau phong rang": "đậu phộng rang",
    "dau nanh": "đậu nành", "dau xanh": "đậu xanh", "dau do": "đậu đỏ",
    "dau den": "đậu đen", "dau trang": "đậu trắng", "dau ga": "đậu gà",
    "dau ngua": "đậu ngự", "dau hoa lan": "đậu hoa lan",
    "hat dieu": "hạt điều", "hat hanh nhan": "hạt hạnh nhân",
    "hat oc cho": "hạt óc chó", "hat thong": "hạt thông",
    "hat me": "hạt mè", "hat me trang": "hạt mè trắng", "hat me den": "hạt mè đen",
    "hat sen": "hạt sen", "hat bap": "hạt bắp", "hat tieu": "hạt tiêu",
    "hat e": "hạt é", "hat chia": "hạt chia", "hat gao": "hạt gạo",
    
    # BÁNH & MÌ
    "banh mi": "bánh mì", "banh mi baguette": "bánh mì baguette",
    "banh mi sandwich": "bánh mì sandwich", "banh mi burger": "bánh mì burger",
    "banh mi hot dog": "bánh mì hot dog", "banh mi trang": "bánh mì trắng",
    "banh mi den": "bánh mì đen", "banh mi o": "bánh mì ổ",
    "banh trang": "bánh tráng", "banh trang me": "bánh tráng mè",
    "banh trang gaol": "bánh tráng gạo", "banh trang bot gao": "bánh tráng bột gạo",
    "banh trang bo bia": "bánh tráng bò bía", "banh trang chao gio": "bánh tráng chả giò",
    "banh phong tom": "bánh phồng tôm", "banh da": "bánh đa",
    "banh da nem": "bánh đa nem", "banh da tom": "bánh đa tôm",
    "banh canh": "bánh canh", "banh canh bot gao": "bánh canh bột gạo",
    "banh canh bot loc": "bánh canh bột lọc",
    "banh bao": "bánh bao", "banh bao chay": "bánh bao chay",
    "banh xeo": "bánh xèo", "banh khot": "bánh khọt",
    "banh cuon": "bánh cuốn", "banh beo": "bánh bèo",
    "banh uot": "bánh ướt", "banh uot khong nhan": "bánh ướt không nhân",
    "banh chung": "bánh chưng", "banh tet": "bánh tét", "banh pia": "bánh pía",
    "banh flan": "bánh flan", "banh gan": "bánh gan",
    "banh quy": "bánh quy", "banh cookie": "bánh cookie",
    "banh gao": "bánh gạo", "banh gao han quoc": "bánh gạo hàn quốc",
    "mi": "mì", "mi tom": "mì tôm", "mi goi": "mì gói",
    "mi han quoc": "mì hàn quốc", "mi udon": "mì udon", "mi ramen": "mì ramen",
    "mi soba": "mì soba", "mi spaghetti": "mì spaghetti", "mi y": "mì ý",
    "mi trung": "mì trứng", "mi xao": "mì xào",
    "bun": "bún", "bun bo": "bún bò", "bun rieu": "bún riêu",
    "bun cha": "bún chả", "bun dau": "bún đậu", "bun tom": "bún tôm",
    "bun mam": "bún mắm", "bun tau": "bún tàu", "bun me den": "bún mè đen",
    "pho": "phở", "pho bo": "phở bò", "pho ga": "phở gà",
    "pho chay": "phở chay", "pho xao": "phở xào",
    "hu tieu": "hủ tiếu", "hu tieu dai": "hủ tiếu dai",
    "hu tieu my tho": "hủ tiếu mỹ tho", "hu tieu nam vang": "hủ tiếu nam vang",
    "miến": "miến", "mien dong": "miến dong", "mien ga": "miến gà",
    
    # NƯỚC & ĐỒ UỐNG
    "nuoc": "nước", "nuoc loc": "nước lọc", "nuoc suoi": "nước suối",
    "nuoc dua": "nước dừa", "nuoc mia": "nước mía", "nuoc cam": "nước cam",
    "nuoc chanh": "nước chanh", "nuoc chanh day": "nước chanh dây",
    "nuoc ep": "nước ép", "nuoc ep tao": "nước ép táo",
    "nuoc ep cam": "nước ép cam", "nuoc ep ca rot": "nước ép cà rốt",
    "nuoc ep dua hau": "nước ép dưa hấu",
    "bia": "bia", "bia hoi": "bia hơi", "bia lon": "bia lon",
    "ruou": "rượu", "ruou vang": "rượu vang", "ruou trang": "rượu trắng",
    "ruou nep": "rượu nếp", "ruou mai que lo": "rượu mai quế lộ",
    "ruou man": "rượu mận", "ruou sake": "rượu sake",
    "tra": "trà", "tra xanh": "trà xanh", "tra dao": "trà đào",
    "tra vai": "trà vải", "tra sen": "trà sen", "tra mat ong": "trà mật ong",
    "tra sua": "trà sữa", "tra sua tran chau": "trà sữa trân châu",
    "ca phe": "cà phê", "ca phe den": "cà phê đen", "ca phe sua": "cà phê sữa",
    "ca phe phin": "cà phê phin", "ca phe hoa tan": "cà phê hòa tan",
    
    # CÁC LOẠI KHÁC
    "com": "cơm", "com trang": "cơm trắng", "com nep": "cơm nếp",
    "com gao lut": "cơm gạo lứt", "com rang": "cơm rang",
    "xoi": "xôi", "xoi nep": "xôi nếp", "xoi lac": "xôi lạc",
    "xoi dau xanh": "xôi đậu xanh", "xoi gac": "xôi gấc",
    "xoi la cam": "xôi lá cẩm", "xoi ngo": "xôi ngô",
    "che": "chè", "che dau xanh": "chè đậu xanh", "che dau do": "chè đậu đỏ",
    "che ba ba": "chè bà ba", "che chuoi": "chè chuối",
    "che hat sen": "chè hạt sen", "che khoai": "chè khoai",
    "kem": "kem", "kem vani": "kem vani", "kem socola": "kem socola",
    "kem dau": "kem dâu", "kem matcha": "kem matcha",
    "thach": "thạch", "thach rau cau": "thạch rau câu",
    "thach dua": "thạch dừa", "thach trai cay": "thạch trái cây",
    "mut": "mứt", "mut dau": "mứt dâu", "mut cam": "mứt cam",
    "mut bi do": "mứt bí đỏ", "mut coc": "mứt cóc",
    
    # TỪ VIẾT TẮT & BIẾN THỂ THÔNG DỤNG
    "tp": "thịt", "tp bo": "thịt bò", "tp heo": "thịt heo",
    "hs": "hải sản", "hs tuoi": "hải sản tươi",
    "rv": "rau củ", "rv qua": "rau củ quả",
    "gv": "gia vị", "gv nem": "gia vị nêm",
    "nc": "nước", "nc mam": "nước mắm", "nc tuong": "nước tương",
    "b": "bột", "b mi": "bột mì", "b nep": "bột nếp",
    "t": "trứng", "t ga": "trứng gà", "t vit": "trứng vịt",
    "s": "sữa", "s tuoi": "sữa tươi", "s chua": "sữa chua",
    "d": "đường", "d cat": "đường cát", "d phen": "đường phèn",
    "m": "muối", "m hat": "muối hạt", "m ot": "muối ớt",

    # THÊM CÁC TỪ TỪ DANH SÁCH CỦA BẠN
    "aji": "", "ajingon": "", "ajiquick": "", "blendy": "",
    "nương": "nướng", "chin": "chín", "cut": "cút", "nuc": "nục",
    "tim": "tím", "rô": "rô", "bao vo": "bào vỏ", "vo": "vỏ",
    "cay": "cay", "lat": "lá", "bam": "bằm", "dap": "đập",
    "luoc": "luộc", "nuong": "nướng", "chien": "chiên", "hap": "hấp",
    "kho": "kho", "xa": "xả", "bung": "bưng", "phan": "phần",
    "giam": "giấm", "cau": "cau", "doli": "doli", "laba": "laba",
    "chat": "chát", "sap": "sáp", "sua": "sứa", "hu": "hủ",
    "ky": "kỳ", "nho": "nhỏ", "cac": "các", "bong": "bông",
    "bong lau": "bông lau", "bong lan": "bông lan", "bong atiso": "bông atiso",
    "bong cai": "bông cải", "bong thien ly": "bông thiên lý",
    "bong dieu dien": "bông điên điển", "bong sung": "bông súng",
    "bong so dua": "bông so đũa", "bong kim cham": "bông kim châm",
    "bong he": "bông hẹ", "bong hanh": "bông hành",
    "bong muop": "bông mướp", "bong bi": "bông bí",
}


def apply_corrections(name: str) -> str:
    if not name:
        return ""

    words = name.split()
    corrected_words = []
    
    for word in words:
        lower_word = word.lower()
        if lower_word in INGREDIENT_CORRECTIONS:
            corrected = INGREDIENT_CORRECTIONS[lower_word]
            if corrected:  # Chỉ thay thế nếu có giá trị
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)  # Giữ nguyên nếu giá trị rỗng
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words).strip()


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

    # 3. Loại bỏ động từ, hành động chế biến
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

    # 4. Loại bỏ thương hiệu
    brand_map = ["aji-ngon", "aji-no-moto", "phú sĩ", "ajinomoto"]
    for b in brand_map:
        raw = raw.replace(b, "")

    # 5. Gom nhóm bằng replacements
    replacements = {
        "hạt nêm ajingon heo": "hạt nêm",
        "hạt nêm ajingon nấm": "hạt nêm",
        "hạt nêm ajingon gà": "hạt nêm",
        "bột ngọt ajinomoto": "bột ngọt",
        "ajinomoto giấm gao len men": "giấm gạo lên men",
        "nước tương phú sĩ": "nước tương",
        "nước tương lisa" : "nước tương",
        "xốt tương đậu nành lisa": "xốt tương đậu nành",
        "xốt mayonnaise ajimayo vị ngọt dịu": "xốt mayonnaise",
        "xốt mayonnaise ajimayo vị nguyên bản": "xốt mayonnaise",
        "ajiquick bột": "bột chiên giòn",
        "ajiquick bột tẩm": "bột chiên giòn",
        "ajiquick bột tẩm khô giòn": "bột chiên giòn",
        "ajiquick bột giòn": "bột chiên giòn",
        "nêm ajiquick lẩu" : "gia vị nêm sẵn lẩu",
        "nêm sẵn ajiquick lẩu" : "gia vị nêm sẵn lẩu",
        "nêm sẵn ajiquick thịt kho" :"gói gia vị nêm sẵn nấu thịt kho",
        "đầu hành và hành tím" : 'hành', 
        "xốt dùng ngay kho quẹt" : "kho quẹt",
        "nêm sẵn ajiquick phở bò" : "gia vị nêm sẵn phở bò",
        "nêm sẵn ajiquick bún riêu cua" : "gia vị nêm sẵn bún riêu cua",
    }
    for k, v in replacements.items():
        # xóa space thừa + lowercase trước khi so sánh
        raw_cmp = re.sub(r'\s+', ' ', raw)
        if k in raw_cmp:
            return v

    # 6. Cleanup khoảng trắng
    raw = re.sub(r'\s+', ' ', raw).strip()
    if not raw:
        return ""

    return raw.lower()


# ========================================================================
# 5. TÁCH NGUYÊN LIỆU → (tên, số lượng)
# ========================================================================

def clean_name(name: str) -> str:
    if not name:
        return "" 
    
    # 1) Remove anything inside parentheses
    name = re.sub(r"\(.*?\)", "", name)

    # 2) Normalize: unicode + lowercase + no accents
    name = normalize_ingredient_name(name)

    # 3) 🔥 ÁP DỤNG SỬA CHÍNH TẢ Ở ĐÂY
    name = apply_corrections(name)

    # 4) Clean spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


def detect_ingredient_parts(text: str) -> Tuple[str, Union[str, None]]:
    """Tách 1 dòng nguyên liệu → (name, qty) chuẩn hóa nâng cao."""
    text = text.strip()

    # --- 1. Tách nếu có nhiều nguyên liệu bằng dấu phẩy (chỉ lấy phần đầu vì vòng for xử lý từng item) ---
    if "," in text:
        text = text.split(",")[0].strip()

    # --- 2. Nếu có dấu ":" tách name : quantity ---
    if ":" in text:
        parts = text.split(":", 1)
        name_part = parts[0]
        qty_part = parts[1] if len(parts) > 1 else ""
        name = clean_name(name_part)
        qty = qty_part.strip() or None

        # 🔥 CHUẨN HÓA ĐƠN VỊ
        if qty:
            qty = normalize_unit(qty)

        return name, qty

    # --- 3. Regex tìm số lượng ---
    match = re.search(r"(\d[\d\s./]*\s*(?:g|gam|kg|ml|trái|cây|muỗng|quả|lá)?)", text, flags=re.I)

    if match:
        quantity = match.group(0).strip() or None
        name = text[:match.start()].strip()
        name = clean_name(name)

        # 🔥 CHUẨN HÓA ĐƠN VỊ
        if quantity:
            quantity = normalize_unit(quantity)

        return name, quantity

    # --- 4. Không tìm thấy số lượng → quantity = None ---
    name = clean_name(text)
    return name, None


def process_ingredients(ingredients: List[str]) -> Tuple[List[str], List[Union[str, None]]]:
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

def detect_category(name: str) -> str:
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
# 7. HÀM CHÍNH XỬ LÝ TOÀN BỘ DATAFRAME - ĐÃ TÍCH HỢP CHUẨN HÓA
# ========================================================================

def process_and_export(raw_data: List[Dict], output_file: str) -> pd.DataFrame:
    df = pd.DataFrame(raw_data) 
    if "dish_name" in df.columns:
        df["dish_name"] = df["dish_name"].apply(normalize_text).apply(normalize_name)
    
    # Xử lý nguyên liệu - ÁP DỤNG CORRECTIONS TRỰC TIẾP
    if "ingredients" in df.columns:
        print("🔧 Đang xử lý nguyên liệu...")
        
        # Áp dụng corrections cho từng ingredient trước
        df["ingredients_corrected"] = df["ingredients"].apply(
            lambda lst: [apply_corrections(str(item)) for item in lst] if isinstance(lst, list) else [apply_corrections(str(lst))]
        )
        
        df["ingredient_names"], df["ingredient_quantities"] = zip(
            *df["ingredients_corrected"].apply(process_ingredients)
        )
        df = df.drop(columns=["ingredients", "ingredients_corrected"])

    if "cooking_time" in df.columns:
        df["cooking_time"] = df["cooking_time"].apply(normalize_cook_time)

    if "servings" in df.columns:
        df["servings"] = df["servings"].apply(normalize_servings)

    if "dish_name" in df.columns:
        df["category"] = df["dish_name"].apply(detect_category)

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])

    df = df.reset_index(drop=True)
    df["index"] = df.index + 1
    
    if "ingredient_names" in df.columns:
        df["ingredient_count"] = df["ingredient_names"].apply(len)
    
    # Xóa khối lượng
    if "ingredient_quantities" in df.columns:
        df = df.drop(columns=["ingredient_quantities"])
        
    

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_json(output_file, orient="records", indent=2, force_ascii=False)
    print("✅ Đã xuất file", output_file)
    
    return df


# ========================================================================
# 8. CHẠY TRỰC TIẾP
# ========================================================================

if __name__ == "__main__":
    input_file = r".\data\recipes.json"
    output_file = r".\data\recipes_501_1000_cleaned.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cleaned = clean_json(raw)
        df_result = process_and_export(cleaned, output_file)

        # Xóa escape \/ trong URL
        with open(output_file, "r", encoding="utf-8") as f:
            data = f.read().replace("\\/", "/")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(data)
        
        # Xuất ra CSV
        csv_output = r"./data/data.csv"
        os.makedirs(os.path.dirname(csv_output), exist_ok=True)
        df_result.to_csv(csv_output, index=False, encoding="utf-8-sig")
        print("🎉 Hoàn tất.")
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {input_file}")
    except json.JSONDecodeError:
        print(f"❌ Lỗi đọc file JSON: {input_file}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")