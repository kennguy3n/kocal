#!/usr/bin/env python3
"""Expand image search dataset to 120+ cases with multilingual queries."""
import json, copy

with open("image_search_eval_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["test_cases"]

# Add language field to existing cases
for tc in cases:
    if "language" not in tc:
        tc["language"] = "en"

# Multilingual image search queries
multilingual_queries = {
    "ja": [
        ("山の風景", "landscape", "山岳風景"),
        ("富士山", "landscape", "富士山"),
        ("桜の花", "portrait", "桜の花"),
        ("日本料理", "square", "日本料理"),
        ("東京の夜景", "landscape", "東京夜景"),
        ("伝統的な日本建築", "landscape", "日本建築"),
        ("猫の写真", "square", "猫"),
        ("海辺の夕日", "landscape", "海辺の夕日"),
    ],
    "ko": [
        ("산 풍경", "landscape", "山岳風景"),
        ("서울 야경", "landscape", "ソウル夜景"),
        ("한식", "square", "韓国料理"),
        ("벚꽃", "portrait", "桜"),
        ("강아지", "square", "子犬"),
        ("바다 일몰", "landscape", "海の夕日"),
    ],
    "zh": [
        ("山景", "landscape", "山の風景"),
        ("长城", "landscape", "長城"),
        ("中国菜", "square", "中華料理"),
        ("熊猫", "square", "パンダ"),
        ("上海夜景", "landscape", "上海夜景"),
        ("花园", "landscape", "庭園"),
    ],
    "es": [
        ("montañas", "landscape", "山脈"),
        ("playa", "landscape", "ビーチ"),
        ("comida mexicana", "square", "メキシコ料理"),
        ("ciudad de noche", "landscape", "夜の都市"),
        ("flores", "portrait", "花"),
        ("gato", "square", "猫"),
    ],
    "vi": [
        ("núi non", "landscape", "山岳風景"),
        ("biển", "landscape", "海"),
        ("phở", "square", "フォー"),
        ("thành phố về đêm", "landscape", "夜の都市"),
        ("hoa", "portrait", "花"),
        ("mèo", "square", "猫"),
    ],
    "fr": [
        ("montagne", "landscape", "山"),
        ("plage", "landscape", "ビーチ"),
        ("cuisine française", "square", "フランス料理"),
        ("Paris la nuit", "landscape", "夜のパリ"),
        ("fleurs", "portrait", "花"),
        ("chat", "square", "猫"),
    ],
    "de": [
        ("Berge", "landscape", "山"),
        ("Strand", "landscape", "ビーチ"),
        ("deutsche Küche", "square", "ドイツ料理"),
        ("Berlin bei Nacht", "landscape", "夜のベルリン"),
        ("Blumen", "portrait", "花"),
        ("Hund", "square", "犬"),
    ],
    "ar": [
        ("جبال", "landscape", "山"),
        ("شاطئ", "landscape", "ビーチ"),
        ("طعام عربي", "square", "アラブ料理"),
        ("مدينة ليلا", "landscape", "夜の都市"),
        ("ورود", "portrait", "花"),
        ("قط", "square", "猫"),
    ],
    "hi": [
        ("पहाड़", "landscape", "山"),
        ("समुद्र तट", "landscape", "ビーチ"),
        ("भारतीय खाना", "square", "インド料理"),
        ("मुंबई रात", "landscape", "夜のムンバイ"),
        ("फूल", "portrait", "花"),
        ("बिल्ली", "square", "猫"),
    ],
    "th": [
        ("ภูเขา", "landscape", "山"),
        ("หาดทราย", "landscape", "ビーチ"),
        ("อาหารไทย", "square", "タイ料理"),
        ("กรุงเทพกลางคืน", "landscape", "夜のバンコク"),
        ("ดอกไม้", "portrait", "花"),
        ("แมว", "square", "猫"),
    ],
    "id": [
        ("gunung", "landscape", "山"),
        ("pantai", "landscape", "ビーチ"),
        ("makanan indonesia", "square", "インドネシア料理"),
        ("jakarta malam", "landscape", "夜のジャカルタ"),
        ("bunga", "portrait", "花"),
        ("kucing", "square", "猫"),
    ],
    "pt": [
        ("montanha", "landscape", "山"),
        ("praia", "landscape", "ビーチ"),
        ("comida brasileira", "square", "ブラジル料理"),
        ("rio de janeiro noite", "landscape", "夜のリオ"),
        ("flores", "portrait", "花"),
        ("gato", "square", "猫"),
    ],
    "tl": [
        ("bundok", "landscape", "山"),
        ("dagat", "landscape", "海"),
        ("pagkaing pilipino", "square", "フィリピン料理"),
        ("maynila gabi", "landscape", "夜のマニラ"),
        ("bulaklak", "portrait", "花"),
        ("pusa", "square", "猫"),
    ],
}

max_num = 0
for tc in cases:
    num = int(tc["id"].replace("img_", ""))
    max_num = max(max_num, num)

counter = max_num
new_cases = []

# Use "any" provider for multilingual queries (registry handles fallback)
for lang, queries in multilingual_queries.items():
    for query, orientation, desc in queries:
        counter += 1
        new_tc = {
            "id": f"img_{counter:03d}",
            "provider": "any",
            "query": query,
            "orientation": orientation,
            "per_page": 5,
            "safesearch": True,
            "expected_min_results": 1,
            "expected_orientation": orientation,
            "description": f"{desc} ({lang})",
            "language": lang,
        }
        new_cases.append(new_tc)

# Add semantic relevance queries (paraphrased)
semantic_queries = [
    ("tall buildings at night", "landscape", "urban skyline"),
    ("green forest from above", "landscape", "aerial forest"),
    ("person working on laptop", "landscape", "remote work"),
    ("delicious breakfast spread", "landscape", "breakfast food"),
    ("calm lake reflection", "landscape", "lake reflection"),
    ("colorful autumn leaves", "portrait", "autumn foliage"),
    ("modern office space", "landscape", "office interior"),
    ("happy diverse team meeting", "landscape", "team collaboration"),
    ("fresh coffee cup morning", "portrait", "coffee"),
    ("mountain hiking trail", "landscape", "hiking"),
]

for query, orientation, desc in semantic_queries:
    counter += 1
    new_tc = {
        "id": f"img_{counter:03d}",
        "provider": "any",
        "query": query,
        "orientation": orientation,
        "per_page": 5,
        "safesearch": True,
        "expected_min_results": 1,
        "expected_orientation": orientation,
        "description": f"Semantic: {desc}",
        "language": "en",
    }
    new_cases.append(new_tc)

cases.extend(new_cases)
print(f"Original cases: {len(cases) - len(new_cases)}")
print(f"New cases: {len(new_cases)}")
print(f"Total cases: {len(cases)}")

data["version"] = "2.0.0"
data["description"] = f"Comprehensive image search evaluation dataset with {len(cases)} cases across 4 providers, including multilingual queries in 14 languages and semantic relevance tests."

with open("image_search_eval_v2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

from collections import Counter
langs = Counter(tc.get("language", "?") for tc in cases)
providers = Counter(tc.get("provider", "?") for tc in cases)
print(f"\nBy language: {dict(langs)}")
print(f"By provider: {dict(providers)}")
print(f"File size: {len(json.dumps(data))} bytes")
