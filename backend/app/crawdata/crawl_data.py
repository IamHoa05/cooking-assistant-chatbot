from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, json, os

def crawl_monngonmoingay(search_url, max_pages=1, scroll_times=3, limit_per_page=50):
    driver = webdriver.Chrome()
    all_recipes = []

    # --- Thu thập link + ảnh từ list pages ---
    for page in range(1, max_pages+1):
        url = f"{search_url}&page={page}"
        driver.get(url)

        # Đóng popup cookie (chỉ lần đầu)
        if page == 1:
            try:
                close_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "span.cookieDrawer__close"))
                )
                close_btn.click()
                time.sleep(1)
            except:
                pass

        # Cuộn xuống để load thêm nội dung
        cards = driver.find_elements(By.CSS_SELECTOR, "div.relative.rounded-xl a")[:limit_per_page]
        for card in cards:
            href = card.get_attribute("href")
            try:
                img = card.find_element(By.TAG_NAME, "img").get_attribute("src")
            except:
                img = ""
            if href and "monngonmoingay.com" in href:
                all_recipes.append({"url": href, "image_link": img})

    # --- Crawl chi tiết từng món ---
    results = []
    for recipe in all_recipes:
        driver.get(recipe["url"])
        time.sleep(1)

        # Tên món
        try:
            title_element = driver.find_element(By.CSS_SELECTOR, "span.title")
            dish_name = title_element.text.strip()
        except:
            dish_name = ""

        # Nguyên liệu
        try:
            ingredients = [el.text.strip() for el in driver.find_elements(By.CSS_SELECTOR, "div.block-nguyenlieu li") if el.text.strip()]
        except:
            ingredients = []

        # Các bước nấu
        try:
            steps = driver.find_elements(By.CSS_SELECTOR, "#section-soche li, #section-soche p, #section-thuchien li, #section-thuchien p")
            instructions = [el.text.strip() for el in steps if el.text.strip()]
        except:
            instructions = []

        # Khẩu phần, thời gian, độ khó
        try:
            info_blocks = driver.find_elements(By.CSS_SELECTOR, "div.flex.flex-col.gap-1.text-sm.items-center")
            servings = info_blocks[0].find_element(By.CSS_SELECTOR, "strong").text.strip()
            cooking_time = info_blocks[1].find_element(By.CSS_SELECTOR, "strong").text.strip()
            difficulty = info_blocks[2].find_element(By.CSS_SELECTOR, "strong").text.strip()
        except:
            servings, cooking_time, difficulty = "NA", "NA", "NA"

        # Cách dùng
        try:
            usage = [el.text.strip() for el in driver.find_elements(By.CSS_SELECTOR, "#section-howtouse p, #section-howtouse li") if el.text.strip()]
        except:
            usage = []

        # Tips
        try:
            tips = [el.text.strip() for el in driver.find_elements(By.CSS_SELECTOR, "#section-tips p, #section-tips li") if el.text.strip()]
        except:
            tips = []

        results.append({
            "url": recipe["url"],
            "dish_name": dish_name,
            "ingredients": ingredients,
            "instructions": instructions,
            "cooking_time": cooking_time,
            "servings": servings,
            "difficulty": difficulty,
            "usage": usage,
            "tips": tips,
            "image_link": recipe["image_link"]
        })

    driver.quit()
    return results



recipes = crawl_monngonmoingay(
    "https://monngonmoingay.com/tim-kiem-mon-ngon/page/5/?tim=1&nguyen-lieu=mon-ngon-tu-thit-ga&sort=like",
    max_pages=2,      
    scroll_times=3,   # scroll 3 lần mỗi trang
    limit_per_page=50 # số món tối đa mỗi trang
)

# Xuất JSON
filename = "recipes.json"
# Đọc dữ liệu cũ nếu file tồn tại
if os.path.exists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        old_data = json.load(f)
else:
    old_data = []

# Nối thêm dữ liệu mới
all_data = old_data + recipes

# Lưu lại
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)