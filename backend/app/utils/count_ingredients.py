import pandas as pd
import ast
from collections import Counter

df = pd.read_csv("./data/recipes_cleaned.csv")

all_ingredients = []

for x in df['ingredient_names']:
    try:
        ings = ast.literal_eval(x)  # chuyển chuỗi thành list
        all_ingredients.extend(ings)
    except:
        continue

counter = Counter(all_ingredients)

# Top 20 nguyên liệu phổ biến
top20 = counter.most_common(20)
print("🔹 Top 20 nguyên liệu phổ biến:")
for ing, cnt in top20:
    print(f"{ing}: {cnt}")

# Lưu toàn bộ danh sách
df_counts = pd.DataFrame(counter.items(), columns=['ingredient', 'count']).sort_values(by='count', ascending=False)
df_counts.to_csv("ingredient_counts.csv", index=False)
print("✅ Đã lưu toàn bộ danh sách nguyên liệu vào ingredient_counts.csv")
