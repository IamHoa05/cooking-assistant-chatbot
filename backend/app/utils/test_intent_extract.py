"""
Test Intent + Slot Extraction
Refactor version dùng detect_intent + extract_all_slots
In kết quả dạng bảng text giống LaTeX mẫu
"""

import random
from sklearn.metrics import precision_score, recall_score, f1_score
from detect_intent_extract import detect_intent, extract_all_slots, format_output_by_intent

# ============================
# DATA DEFINITIONS
# ============================
ingredients = ["thịt heo", "thịt bò", "gà", "vịt", "cá hồi", "cá basa", "tôm", "mực",
               "trứng", "khoai tây", "cà rốt", "bí đỏ", "đậu hũ", "nấm", "bắp cải",
               "cà chua", "hành tây", "tỏi", "gừng", "rau muống", "rau cải", "bông cải xanh"]

dishes = ["canh chua chem chép", "nấm đùi gà kho gừng", "bò xào hành tây", "tôm mực chiên giòn", "cá hồi ngũ sắc",
          "súp bí đỏ", "bún bò giò heo", "phở gà", "cháo tôm nếp cẩm", "bánh flan", "bánh mì tẩm trứng chiên", 
          "lagu bò", "bì bún chay", "cơm chiên dương châu", "canh rau muống khoai sọ", "canh boaro nấu đậu hũ non",
          "salad trộn kiểu nhật", "canh gà bí xanh", "cá rô kho tộ", "khoai tây chiên bò nướng"]

categories = ["món chay", "ít calo", "món nước", "món chiên", "món kho", "ăn kiêng"]
intents = ["cooking_guide", "suggest_dishes"]

templates_recommend = [
    "Tôi thèm {ing}, gợi ý món nào nấu trong {time} phút?",
    "Tôi vừa mua {ing}, có món nào ngon không?",
    "Cho tôi vài món làm từ {ing} nhưng thời gian nấu dưới {time} phút.",
    "Tôi muốn nấu món từ {ing1} và {ing2}, miễn là không quá {time} phút.",
    "Tôi đang giảm cân, cho tôi món {cat} từ {ing}.",
    "Có món nào từ {ing} mà trẻ con thích không?",
    "Tôi bị dị ứng với hải sản, nên hãy gợi ý món từ {ing}.",
    "Hôm nay trời lạnh quá, có món nóng nào với {ing} không?",
    "Tôi có {ing1} và {ing2}, làm món gì ngon nhỉ?",
    "Gợi ý món ăn nhanh với {ing} trong vòng {time} phút.",
    "Có món ăn ít calo từ {ing} không?",
    "Tối nay tôi muốn nấu {cat}, có công thức nào đơn giản?",
    "Tôi muốn món {cat} từ {ing1} và {ing2}, bạn gợi ý được không?",
    "Cho tôi vài món ăn phù hợp với {ing} và {ing2}.",
    "Có món {cat} nào dễ làm từ {ing} không?",
    "Món nào ngon với {ing} mà không tốn quá {time} phút?",
     "Tôi có {ing1}, {ing2} và {ing3}, nấu món gì ngon nhỉ?",
    "Bạn có thể gợi ý vài món {cat} từ {ing} không?",
    "Tôi muốn nấu món {cat}, nhưng chỉ có {ing} và {ing2}.",
    "Có món gì nhanh chóng với {ing} mà tôi có thể nấu trong {time} phút?",
    "Tôi đang muốn thử món {cat}, có công thức nào dễ làm không?",
    "Cho tôi một vài món ăn từ {ing} và {ing2}, phù hợp cho trẻ em.",
    "Tối nay nấu {ing1} và {ing2}, gợi ý món gì dễ làm?",
    "Có món {cat} nào vừa ngon vừa nhanh với {ing} không?",
    "Món ăn từ {ing} nhưng ít dầu mỡ, có thể gợi ý không?",
    "Tôi muốn nấu món {cat} cho {serv} người, bạn có gợi ý không?",
    "Cho tôi vài món từ {ing1}, {ing2}, {ing3} nhưng không quá {time} phút.",
]

templates_guide = [
    "Hướng dẫn tôi cách nấu món {dish} trong {time} phút.",
    "Chỉ tôi cách làm {dish} cho {serv} người.",
    "Làm sao để nấu {dish} cho đúng vị?",
    "Cho tôi công thức nấu món {dish}.",
    "Tôi muốn học nấu {dish}, chỉ tôi cách làm?",
]

# ============================
# GENERATE TEST CASES
# ============================
def generate_test_cases(n=100):
    dataset = []
    for _ in range(n):
        intent_type = random.choice(intents)

        if intent_type == "suggest_dishes":
            template = random.choice(templates_recommend)
            # Khởi tạo đầy đủ các biến
            ing = random.choice(ingredients)
            ing1 = random.choice(ingredients)
            ing2 = random.choice(ingredients)
            ing3 = random.choice(ingredients)
            time = random.choice([10, 15, 20, 25, 30, 40, 45])
            cat = random.choice(categories)
            serv = random.choice([1,2,3,4,5])  # một số template có {serv}

            # Build dict chỉ chứa key xuất hiện
            fmt_dict = {}
            for key in ["ing","ing1","ing2","ing3","time","cat","serv"]:
                if f"{{{key}}}" in template:
                    fmt_dict[key] = locals()[key]

            text = template.format(**fmt_dict)

            # Build slots
            slots = {}
            ingredients_used = [v for k,v in fmt_dict.items() if k.startswith("ing")]
            if ingredients_used:
                slots["ingredient"] = list(set(ingredients_used))
            if "time" in fmt_dict:
                slots["time"] = f"{fmt_dict['time']} phút"
            if "cat" in fmt_dict:
                slots["category"] = fmt_dict["cat"]
            if "serv" in fmt_dict:
                slots["serving"] = str(fmt_dict["serv"])

        else:  # cooking_guide
            template = random.choice(templates_guide)
            dish = random.choice(dishes)
            time = random.choice([15, 20, 30, 45])
            serv = random.choice([1,2,3,4,5])

            fmt_dict = {}
            for key in ["dish","time","serv"]:
                if f"{{{key}}}" in template:
                    fmt_dict[key] = locals()[key]

            text = template.format(**fmt_dict)

            slots = {"dish_name": dish}
            if "time" in fmt_dict:
                slots["time"] = f"{fmt_dict['time']} phút"
            if "serv" in fmt_dict:
                slots["serving"] = str(fmt_dict["serv"])

        dataset.append({"input": text, "intent": intent_type, "slots": slots})
    return dataset

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    test_dataset = generate_test_cases(100)

    # Đếm intent
    intent_counts = {intent:0 for intent in intents}
    intent_correct_counts = {intent:0 for intent in intents}

    # Slot metrics
    slot_precision_list = []
    slot_recall_list = []
    slot_f1_list = []

    for item in test_dataset:
        text = item["input"]
        true_intent = item["intent"]
        true_slots = item["slots"]

        pred_intent, _, _ = detect_intent(text)
        pred_slots = extract_all_slots(text, intent=pred_intent)
        pred_slots_formatted = format_output_by_intent(pred_intent, pred_slots)

        # Intent count
        intent_counts[true_intent] += 1
        if pred_intent == true_intent:
            intent_correct_counts[true_intent] += 1

        # Slot scoring
        true_set = set()
        pred_set = set()

        for k,v in true_slots.items():
            if isinstance(v,list):
                for x in v:
                    true_set.add(f"{k}:{x}")
            else:
                true_set.add(f"{k}:{v}")

        for k,v in pred_slots_formatted.items():
            if isinstance(v,list):
                for x in v:
                    pred_set.add(f"{k}:{x}")
            else:
                pred_set.add(f"{k}:{v}")

        all_items = list(true_set.union(pred_set))
        y_true = [1 if x in true_set else 0 for x in all_items]
        y_pred = [1 if x in pred_set else 0 for x in all_items]

        if len(all_items) > 0:
            slot_precision_list.append(precision_score(y_true, y_pred))
            slot_recall_list.append(recall_score(y_true, y_pred))
            slot_f1_list.append(f1_score(y_true, y_pred))

    # ============================
    # In kết quả dạng bảng text
    # ============================
    # Intent table
    print("\n========= KẾT QUẢ INTENT =========")
    print(f"{'Intent':<20} | {'Số câu đúng / Tổng số':<20} | {'Tỷ lệ chính xác'}")
    print("-"*60)
    total_correct = 0
    total_count = 0
    for intent in intents:
        correct = intent_correct_counts[intent]
        total = intent_counts[intent]
        acc = correct/total*100 if total>0 else 0
        total_correct += correct
        total_count += total
        name = "Hướng dẫn nấu ăn" if intent=="cooking_guide" else "Gợi ý món ăn"
        print(f"{name:<20} | {correct} / {total:<20} | {acc:.0f}%")
    # Trung bình
    avg_acc = total_correct/total_count*100 if total_count>0 else 0
    print(f"{'Trung bình':<20} | {total_correct} / {total_count:<20} | {avg_acc:.0f}%")

    # Slot table
    print("\n========= KẾT QUẢ SLOT =========")
    print(f"{'Slot':<20} | {'Số câu đúng / Tổng số':<20} | {'Tỷ lệ chính xác'}")
    print("-"*60)
    # Các slot chính: dish_name, ingredient
    # Tổng hợp số câu đúng bằng precision * số mẫu tương ứng
    slot_names = [("dish_name", "Tên món ăn"), ("ingredient","Nguyên liệu")]
    for slot_key, slot_label in slot_names:
        count = sum(1 for item in test_dataset if slot_key in item["slots"])
        correct = 0
        for i,item in enumerate(test_dataset):
            if slot_key in item["slots"]:
                # precision_score theo item
                text = item["input"]
                pred_intent, _, _ = detect_intent(text)
                pred_slots = extract_all_slots(text, intent=pred_intent)
                pred_slots_formatted = format_output_by_intent(pred_intent, pred_slots)
                true_val = item["slots"][slot_key]
                pred_val = pred_slots_formatted.get(slot_key, [] if isinstance(true_val,list) else "")
                if isinstance(true_val,list):
                    if set(true_val)==set(pred_val):
                        correct +=1
                else:
                    if true_val==pred_val:
                        correct +=1
        acc = correct/count*100 if count>0 else 0
        print(f"{slot_label:<20} | {correct} / {count:<20} | {acc:.0f}%")
    # Trung bình
    total_correct_slot = sum([sum(1 for i,item in enumerate(test_dataset)
                                  if slot_key in item["slots"] and (
                                      (isinstance(item["slots"][slot_key],list) and
                                       set(item["slots"][slot_key])==set(format_output_by_intent(*detect_intent(item["input"])[:1], extract_all_slots(item["input"], detect_intent(item["input"])[0]))[slot_key])
                                       ) or (item["slots"][slot_key]==format_output_by_intent(*detect_intent(item["input"])[:1], extract_all_slots(item["input"], detect_intent(item["input"])[0]))[slot_key])
                                  )
                                 ) for slot_key,_ in slot_names])
    total_count_slot = sum([sum(1 for item in test_dataset if slot_key in item["slots"]) for slot_key,_ in slot_names])
    avg_slot_acc = total_correct_slot/total_count_slot*100 if total_count_slot>0 else 0
    print(f"{'Trung bình':<20} | {total_correct_slot} / {total_count_slot:<20} | {avg_slot_acc:.0f}%")
