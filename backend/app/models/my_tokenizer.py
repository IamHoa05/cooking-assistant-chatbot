from pyvi import ViTokenizer
import re
from typing import List, Set, Dict, Tuple

class MyTokenizer:
    def __init__(self, custom_dict_path: str = "ingredient_dict.txt"):
        self.food_ngrams: Dict[str, int] = {}  # phrase -> word_count
        self._load_dictionary(custom_dict_path)
        
    def _load_dictionary(self, dict_path: str):
        """Tải từ điển với độ ưu tiên theo độ dài"""
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    phrase = line.strip().lower()
                    if phrase:
                        word_count = len(phrase.split())
                        self.food_ngrams[phrase] = word_count
                        
            #print(f"✅ Đã tải {len(self.food_ngrams)} cụm từ food")
            
        except FileNotFoundError:
            print(f"⚠ Không tìm thấy từ điển: {dict_path}")

    def _protect_food_phrases(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Bảo vệ cụm từ food khỏi bị tokenize sai
        Trả về (protected_text, mapping)
        """
        text_lower = text.lower()
        mapping = {}
        
        # Sắp xếp từ dài đến ngắn (ưu tiên cụm từ dài trước)
        sorted_phrases = sorted(self.food_ngrams.keys(), 
                              key=lambda x: (-len(x), x))
        
        for i, phrase in enumerate(sorted_phrases):
            if phrase in text_lower:
                # Tạo placeholder
                placeholder = f" __FOOD_PHRASE_{i}__ "
                
                # Thay thế toàn bộ xuất hiện
                text_lower = text_lower.replace(phrase, placeholder)
                mapping[placeholder.strip()] = phrase
                
        return text_lower, mapping

    def tokenize(self, text: str) -> str:
        """Tokenize thông minh với food domain"""
        if not text.strip():
            return text
            
        # Bước 1: Bảo vệ cụm từ food
        protected_text, food_mapping = self._protect_food_phrases(text)
        
        # Bước 2: Dùng pyvi tokenize
        try:
            # Giữ nguyên case gốc cho tokenization
            pyvi_input = protected_text
            pyvi_result = ViTokenizer.tokenize(pyvi_input)
        except Exception as e:
            print(f"PyVI error: {e}, using fallback")
            pyvi_result = protected_text
        
        # Bước 3: Khôi phục cụm từ food (với định dạng tokenized)
        final_result = pyvi_result
        for placeholder, original_phrase in food_mapping.items():
            # Chuyển trứng gà thành trứng_gà
            tokenized_phrase = original_phrase.replace(' ', '_')
            final_result = final_result.replace(placeholder, tokenized_phrase)
        
        # Bước 4: Fix spacing và punctuation
        final_result = re.sub(r'\s+', ' ', final_result)
        final_result = re.sub(r'\s+([.,!?])', r'\1', final_result)
        
        return final_result.strip()

    def tokenize_return_list(self, text: str) -> List[str]:
        """
        Trả về list tokens - METHOD NÀY ĐÃ ĐƯỢC THÊM
        """
        tokenized_text = self.tokenize(text)
        # Tách tokens, giữ nguyên các token có dấu gạch dưới
        tokens = []
        current_token = []
        
        for char in tokenized_text:
            if char.isspace():
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
            else:
                current_token.append(char)
        
        if current_token:
            tokens.append(''.join(current_token))
            
        return tokens

    def analyze_text(self, text: str) -> Dict:
        """Phân tích text để debug"""
        tokens = self.tokenize_return_list(text)
        protected_text, mapping = self._protect_food_phrases(text)
        
        return {
            'original': text,
            'protected': protected_text,
            'food_phrases_found': list(mapping.values()),
            'tokens': tokens,
            'tokenized': self.tokenize(text)
        }
    
    def extract_ingredients_only(self, text: str) -> List[str]:
        """
        Trả về các nguyên liệu từ text đã được tokenize
        """
        tokenized_text = self.tokenize(text)
        tokens = tokenized_text.split(' ')
        
        # Lọc ingredients: có dấu gạch dưới HOẶC là từ đơn trong từ điển
        ingredients = []
        for token in tokens:
            clean_token = token.strip(' ,.!?;')  # Loại bỏ dấu câu
            
            # Nếu là từ ghép có dấu gạch dưới
            if '_' in clean_token:
                ingredients.append(clean_token)
            # Hoặc là từ đơn có trong từ điển
            elif clean_token in self.food_ngrams:
                ingredients.append(clean_token)
        
        return ingredients


# Sử dụng
if __name__ == "__main__":
    tokenizer = MyTokenizer("ingredient_dict.txt")

    text = "Tôi có thịt gà, khoai tây, cà rốt và hành tây, tỏi, lê. Tôi nên làm món gì"
    result = tokenizer.extract_ingredients_only(text)
    print(f"Ingredients from User's input: {result}")

