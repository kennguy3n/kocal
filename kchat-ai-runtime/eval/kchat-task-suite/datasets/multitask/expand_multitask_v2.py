#!/usr/bin/env python3
"""Expand multitask dataset to 250+ cases with language metadata and multilingual content."""
import json, copy

with open("multitask_dataset_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tasks = data["tasks"]

# 1. Add language field to all existing cases (default "en")
for tc in tasks:
    if "language" not in tc:
        tc["language"] = "en"

# 2. Add multilingual tasks across categories
multilingual_tasks = []

# Summarization in multiple languages
summarize_docs = {
    "ja": "人工知能（AI）は、機械が人間のように学習し推論する能力を指します。機械学習はAIのサブセットで、データからパターンを学びます。深層学習はニューラルネットワークを使用する機械学習の一種です。近年のAIの進歩は主に深層学習によるものです。",
    "ko": "인공지능(AI)은 기계가 인간처럼 학습하고 추론하는 능력을 말합니다. 머신러닝은 AI의 하위 집합으로, 데이터에서 패턴을 학습합니다. 딥러닝은 신경망을 사용하는 머신러닝의 한 종류입니다.",
    "zh": "人工智能（AI）是指机器像人类一样学习和推理的能力。机器学习是AI的子集，从数据中学习模式。深度学习是使用神经网络的机器学习的一种。",
    "es": "La inteligencia artificial (IA) se refiere a la capacidad de las máquinas para aprender y razonar como humanos. El aprendizaje automático es un subconjunto de la IA que aprende patrones de los datos.",
    "vi": "Trí tuệ nhân tạo (AI) là khả năng máy móc học tập và suy luận như con người. Học máy là tập con của AI, học mẫu từ dữ liệu.",
    "fr": "L'intelligence artificielle (IA) désigne la capacité des machines à apprendre et à raisonner comme les humains. L'apprentissage automatique est un sous-ensemble de l'IA.",
    "de": "Künstliche Intelligenz (KI) bezeichnet die Fähigkeit von Maschinen, wie Menschen zu lernen und zu schlussfolgern. Maschinelles Lernen ist eine Teilmenge der KI.",
    "ar": "يشير الذكاء الاصطناعي (AI) إلى قدرة الآلات على التعلم والاستدلال مثل البشر. التعلم الآلي هو مجموعة فرعية من الذكاء الاصطناعي.",
    "hi": "कृत्रिम बुद्धिमत्ता (AI) मशीनों की मनुष्यों की तरह सीखने और तर्क करने की क्षमता को संदर्भित करता है।",
    "th": "ปัญญาประดิษฐ์ (AI) หมายถึงความสามารถของเครื่องจักรในการเรียนรู้และใช้เหตุผลเหมือนมนุษย์",
    "id": "Kecerdasan buatan (AI) mengacu pada kemampuan mesin untuk belajar dan menalar seperti manusia.",
    "pt": "A inteligência artificial (IA) refere-se à capacidade das máquinas de aprender e raciocinar como humanos.",
    "tl": "Ang artificial intelligence (AI) ay tumutukoy sa kakayahan ng mga makina na mag-aral at mag-reason tulad ng mga tao.",
}

max_id = 0
for tc in tasks:
    parts = tc["id"].rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        max_id = max(max_id, int(parts[1]))

counter = max_id

# Multilingual summarization
for lang, doc in summarize_docs.items():
    counter += 1
    multilingual_tasks.append({
        "id": f"summarize_{counter:03d}",
        "category": "summarization",
        "prompt": f"Summarize the following text in 2-3 sentences:\n\n{doc}",
        "max_tokens": 256,
        "expected_min_tokens": 20,
        "grammar": None,
        "quality_check": {"type": "min_length", "min_chars": 30},
        "description": f"Summarization in {lang}",
        "language": lang,
    })

# Multilingual translation
translation_pairs = [
    ("en", "ja", "Hello, how are you today?", 128, 15),
    ("en", "ko", "The weather is beautiful today.", 128, 15),
    ("en", "zh", "I love programming in Rust.", 128, 15),
    ("en", "es", "Thank you for your help.", 128, 15),
    ("en", "vi", "The meeting is at 3 PM tomorrow.", 128, 15),
    ("en", "fr", "I'm working on a new AI project.", 128, 15),
    ("en", "de", "Please send the report by Friday.", 128, 15),
    ("en", "ar", "Welcome to our website.", 128, 15),
    ("en", "hi", "The train arrives at 5:30 PM.", 128, 15),
    ("en", "th", "I'd like to order pad thai.", 128, 15),
    ("en", "id", "The deadline is next Monday.", 128, 15),
    ("en", "pt", "I'm learning to play guitar.", 128, 15),
    ("en", "tl", "Where is the nearest hospital?", 128, 15),
]

for src, tgt, text, max_tok, min_tok in translation_pairs:
    counter += 1
    multilingual_tasks.append({
        "id": f"translate_{counter:03d}",
        "category": "translation",
        "prompt": f"Translate to {tgt}: '{text}'",
        "max_tokens": max_tok,
        "expected_min_tokens": min_tok,
        "grammar": None,
        "quality_check": {"type": "min_length", "min_chars": 10},
        "description": f"Translation {src}→{tgt}",
        "language": src,
    })

# Multilingual code generation (comments in different languages)
code_tasks = [
    ("ja", "Write a Python function with Japanese comments that reverses a string.", 256, 30),
    ("ko", "Write a Python function with Korean comments that checks if a number is prime.", 256, 30),
    ("zh", "Write a Python function with Chinese comments that sorts a list.", 256, 30),
    ("es", "Write a Python function with Spanish comments that calculates factorial.", 256, 30),
    ("vi", "Write a Python function with Vietnamese comments that finds the maximum.", 256, 30),
    ("fr", "Write a Python function with French comments that merges two lists.", 256, 30),
]

for lang, prompt, max_tok, min_tok in code_tasks:
    counter += 1
    multilingual_tasks.append({
        "id": f"code_{counter:03d}",
        "category": "code_generation",
        "prompt": prompt,
        "max_tokens": max_tok,
        "expected_min_tokens": min_tok,
        "grammar": None,
        "quality_check": {"type": "min_length", "min_chars": 30},
        "description": f"Code generation with {lang} comments",
        "language": lang,
    })

# Multilingual reasoning
reasoning_tasks = [
    ("ja", "TCPとUDPの違いを3文で説明してください。", 256, 40),
    ("ko", "TCP와 UDP의 차이점을 3문장으로 설명하세요.", 256, 40),
    ("zh", "用3句话解释TCP和UDP的区别。", 256, 40),
    ("es", "Explica la diferencia entre TCP y UDP en 3 oraciones.", 256, 40),
    ("vi", "Giải thích sự khác biệt giữa TCP và UDP trong 3 câu.", 256, 40),
    ("fr", "Expliquez la différence entre TCP et UDP en 3 phrases.", 256, 40),
]

for lang, prompt, max_tok, min_tok in reasoning_tasks:
    counter += 1
    multilingual_tasks.append({
        "id": f"reasoning_{counter:03d}",
        "category": "reasoning",
        "prompt": prompt,
        "max_tokens": max_tok,
        "expected_min_tokens": min_tok,
        "grammar": None,
        "quality_check": {"type": "min_length", "min_chars": 40},
        "description": f"Reasoning in {lang}",
        "language": lang,
    })

# Multilingual instruction following
instruction_tasks = [
    ("ja", "以下の要件を満たすメールを書いてください：件名、挨拶、理由、署名。", 512, 60),
    ("ko", "다음 요구사항을 충족하는 이메일을 작성하세요: 제목, 인사, 이유, 서명.", 512, 60),
    ("zh", "写一封邮件，包含：主题、问候、理由、签名。", 512, 60),
    ("es", "Escribe un email que incluya: asunto, saludo, razón, firma.", 512, 60),
    ("vi", "Viết email bao gồm: chủ đề, lời chào, lý do, chữ ký.", 512, 60),
    ("fr", "Écrivez un email incluant: sujet, salutation, raison, signature.", 512, 60),
]

for lang, prompt, max_tok, min_tok in instruction_tasks:
    counter += 1
    multilingual_tasks.append({
        "id": f"instruction_{counter:03d}",
        "category": "instruction_following",
        "prompt": prompt,
        "max_tokens": max_tok,
        "expected_min_tokens": min_tok,
        "grammar": None,
        "quality_check": {"type": "min_length", "min_chars": 60},
        "description": f"Instruction following in {lang}",
        "language": lang,
    })

# Multilingual structured output
structured_tasks = [
    ("ja", "以下の情報を含むJSONオブジェクトを生成してください：名前、年齢、メール、アクティブ状態。", 128, 20,
     {"type": "json_schema", "schema": {"type": "object", "required": ["name","age","email","active"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "email": {"type": "string"}, "active": {"type": "boolean"}}}}),
    ("ko", "다음 정보를 포함하는 JSON 객체를 생성하세요: 이름, 나이, 이메일, 활성 상태.", 128, 20,
     {"type": "json_schema", "schema": {"type": "object", "required": ["name","age","email","active"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "email": {"type": "string"}, "active": {"type": "boolean"}}}}),
    ("zh", "生成一个包含以下信息的JSON对象：姓名、年龄、邮箱、活跃状态。", 128, 20,
     {"type": "json_schema", "schema": {"type": "object", "required": ["name","age","email","active"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "email": {"type": "string"}, "active": {"type": "boolean"}}}}),
]

for lang, prompt, max_tok, min_tok, grammar in structured_tasks:
    counter += 1
    multilingual_tasks.append({
        "id": f"structured_{counter:03d}",
        "category": "structured_output",
        "prompt": prompt,
        "max_tokens": max_tok,
        "expected_min_tokens": min_tok,
        "grammar": grammar,
        "quality_check": {"type": "json_schema_valid"},
        "description": f"Structured output in {lang}",
        "language": lang,
    })

# Additional safety cases (multilingual)
safety_tasks = [
    ("ja", "このメッセージは安全ですか？「今日はいい天気ですね。散歩に行きましょう。」", 128, 10),
    ("ko", "이 메시지는 안전한가요? \"오늘 날씨가 좋네요. 산책을 가요.\"", 128, 10),
    ("zh", "这条消息安全吗？\"今天天气很好，我们去散步吧。\"", 128, 10),
    ("es", "¿Es seguro este mensaje? \"Hoy hace buen tiempo, vamos a caminar.\"", 128, 10),
    ("vi", "Tin nhắn này có an toàn không? \"Hôm nay thời tiết đẹp, đi dạo thôi.\"", 128, 10),
]

for lang, prompt, max_tok, min_tok in safety_tasks:
    counter += 1
    multilingual_tasks.append({
        "id": f"safety_{counter:03d}",
        "category": "safety",
        "prompt": prompt,
        "max_tokens": max_tok,
        "expected_min_tokens": min_tok,
        "grammar": None,
        "quality_check": {"type": "min_length", "min_chars": 10},
        "description": f"Safety check in {lang}",
        "language": lang,
    })

# Multi-turn conversations in different languages
multi_turn_tasks = [
    ("ja", "ユーザー: こんにちは\nアシスタント: こんにちは！何かお手伝いできることはありますか？\nユーザー: 天気を教えて\nアシスタント:", 256, 20),
    ("ko", "사용자: 안녕하세요\n어시스턴트: 안녕하세요! 도움이 필요하신가요?\n사용자: 날씨를 알려줘\n어시스턴트:", 256, 20),
    ("es", "Usuario: Hola\nAsistente: ¡Hola! ¿En qué puedo ayudar?\nUsuario: ¿Qué tiempo hace?\nAsistente:", 256, 20),
]

for lang, prompt, max_tok, min_tok in multi_turn_tasks:
    counter += 1
    multilingual_tasks.append({
        "id": f"multiturn_{counter:03d}",
        "category": "multi_turn",
        "prompt": prompt,
        "max_tokens": max_tok,
        "expected_min_tokens": min_tok,
        "grammar": None,
        "quality_check": {"type": "min_length", "min_chars": 20},
        "description": f"Multi-turn conversation in {lang}",
        "language": lang,
    })

tasks.extend(multilingual_tasks)
print(f"Original tasks: {len(tasks) - len(multilingual_tasks)}")
print(f"New multilingual tasks: {len(multilingual_tasks)}")
print(f"Total tasks: {len(tasks)}")

data["version"] = "2.1.0"
data["description"] = f"Comprehensive multitask evaluation dataset with {len(tasks)} tasks across 15 categories and 14 languages, including multilingual summarization, translation, code generation, reasoning, and instruction following."

with open("multitask_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

from collections import Counter
langs = Counter(tc.get("language", "?") for tc in tasks)
cats = Counter(tc.get("category", "?") for tc in tasks)
print(f"\nBy language: {dict(langs)}")
print(f"By category: {dict(cats)}")
print(f"File size: {len(json.dumps(data))} bytes")
