#!/usr/bin/env python3
"""Generate generation_dataset_v2.json with 40+ prompts across categories and languages."""
import json

prompts = []
pid = 0

def p(prompt, max_tokens, grammar, min_tokens, desc, lang="en", category="general"):
    global pid
    pid += 1
    prompts.append({
        "id": f"gen_{pid:03d}",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "grammar": grammar,
        "expected_min_tokens": min_tokens,
        "description": desc,
        "language": lang,
        "category": category,
    })

# --- Greeting / Short text ---
p("Write a short greeting message to welcome a new team member.", 128, None, 10, "Simple greeting generation", "en", "greeting")
p("新しいチームメンバーを歓迎する短いメッセージを書いてください。", 128, None, 10, "Japanese greeting", "ja", "greeting")
p("Escribe un mensaje de bienvenida corto para un nuevo miembro del equipo.", 128, None, 10, "Spanish greeting", "es", "greeting")
p("Viết một tin nhắn chào mừng ngắn gọn cho thành viên mới của nhóm.", 128, None, 10, "Vietnamese greeting", "vi", "greeting")
p("팀의 새 멤버를 환영하는 짧은 인사 메시지를 작성하세요.", 128, None, 10, "Korean greeting", "ko", "greeting")

# --- Summarization ---
p("Summarize the following text in 2-3 sentences: 'The Rust programming language was originally designed by Graydon Hoare at Mozilla Research in 2010. It emphasizes memory safety without garbage collection, using a borrow checker to enforce ownership rules at compile time. Rust 1.0 was released in 2015 and has since been adopted by major companies including Google, Microsoft, and Amazon for systems programming.'", 256, None, 30, "Summarization task", "en", "summarization")
p("次のテキストを2-3文で要約してください：'人工知能（AI）は、機械が人間のように学習し推論する能力を指します。機械学習はAIのサブセットで、データからパターンを学びます。深層学習はニューラルネットワークを使用する機械学習の一種です。近年のAIの進歩は主に深層学習によるものです。'", 256, None, 30, "Japanese summarization", "ja", "summarization")
p("Resume el siguiente texto en 2-3 oraciones: 'La nube computacional permite el acceso bajo demanda a recursos informáticos a través de internet. Los principales proveedores incluyen AWS, Google Cloud y Microsoft Azure. La nube ofrece escalabilidad, pago por uso y eliminación de la gestión de infraestructura física.'", 256, None, 30, "Spanish summarization", "es", "summarization")
p("Tóm tắt đoạn văn sau trong 2-3 câu: 'Trí tuệ nhân tạo (AI) là khả năng máy móc học tập và suy luận như con người. Học máy là tập con của AI, học mẫu từ dữ liệu. Học sâu sử dụng mạng nơ-ron. Tiến bộ AI gần đây chủ yếu nhờ học sâu.'", 256, None, 30, "Vietnamese summarization", "vi", "summarization")

# --- JSON Schema constrained ---
p("Generate a user profile with name, age, email, and active status.", 128,
  {"type": "json_schema", "schema": {"type": "object", "required": ["name","age","email","active"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "email": {"type": "string"}, "active": {"type": "boolean"}}}},
  20, "JSON Schema constrained generation", "en", "json_schema")

p("Create a todo list with 3 items, each having id, title, and completed status.", 256,
  {"type": "json_schema", "schema": {"type": "array", "items": {"type": "object", "required": ["id","title","completed"], "properties": {"id": {"type": "number"}, "title": {"type": "string"}, "completed": {"type": "boolean"}}}}},
  30, "JSON array with schema constraint", "en", "json_schema")

p("Generate a meeting event with title, start_time, duration_minutes, and attendees list.", 256,
  {"type": "json_schema", "schema": {"type": "object", "required": ["title","start_time","duration_minutes","attendees"], "properties": {"title": {"type": "string"}, "start_time": {"type": "string"}, "duration_minutes": {"type": "number"}, "attendees": {"type": "array", "items": {"type": "string"}}}}},
  30, "Complex JSON Schema with nested array", "en", "json_schema")

p("Create a product listing with name, price, currency, in_stock, and tags array.", 256,
  {"type": "json_schema", "schema": {"type": "object", "required": ["name","price","currency","in_stock","tags"], "properties": {"name": {"type": "string"}, "price": {"type": "number"}, "currency": {"type": "string"}, "in_stock": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}}}},
  30, "Product listing JSON Schema", "en", "json_schema")

p("Generate a weather report with location, temperature, unit, conditions, and humidity.", 256,
  {"type": "json_schema", "schema": {"type": "object", "required": ["location","temperature","unit","conditions","humidity"], "properties": {"location": {"type": "string"}, "temperature": {"type": "number"}, "unit": {"type": "string"}, "conditions": {"type": "string"}, "humidity": {"type": "number"}}}},
  30, "Weather report JSON Schema", "en", "json_schema")

p("Create a restaurant review with restaurant_name, rating, cuisine, review_text, and would_recommend.", 256,
  {"type": "json_schema", "schema": {"type": "object", "required": ["restaurant_name","rating","cuisine","review_text","would_recommend"], "properties": {"restaurant_name": {"type": "string"}, "rating": {"type": "number"}, "cuisine": {"type": "string"}, "review_text": {"type": "string"}, "would_recommend": {"type": "boolean"}}}},
  30, "Restaurant review JSON Schema", "en", "json_schema")

p("Generate a task plan with project, priority, assignee, due_date, and subtasks array.", 256,
  {"type": "json_schema", "schema": {"type": "object", "required": ["project","priority","assignee","due_date","subtasks"], "properties": {"project": {"type": "string"}, "priority": {"type": "string"}, "assignee": {"type": "string"}, "due_date": {"type": "string"}, "subtasks": {"type": "array", "items": {"type": "object", "required": ["title","done"], "properties": {"title": {"type": "string"}, "done": {"type": "boolean"}}}}}}},
  40, "Nested JSON Schema with subtasks", "en", "json_schema")

# --- Translation ---
p("Translate to Japanese: 'Hello, how are you today? I hope you're having a wonderful day.'", 128, None, 15, "English to Japanese translation", "en", "translation")
p("Translate to Spanish: 'The weather is beautiful today. Let's go for a walk in the park.'", 128, None, 15, "English to Spanish translation", "en", "translation")
p("Translate to Vietnamese: 'I love programming in Rust because it's safe and fast.'", 128, None, 15, "English to Vietnamese translation", "en", "translation")
p("Translate to Korean: 'Thank you for your help. I really appreciate your support.'", 128, None, 15, "English to Korean translation", "en", "translation")
p("Translate to Chinese: 'The meeting is scheduled for 3 PM tomorrow in the main conference room.'", 128, None, 15, "English to Chinese translation", "en", "translation")
p("Translate to French: 'I'm working on a new project that uses artificial intelligence.'", 128, None, 15, "English to French translation", "en", "translation")
p("Translate to German: 'Please send me the report by Friday afternoon.'", 128, None, 15, "English to German translation", "en", "translation")
p("Translate to Arabic: 'Welcome to our website. How can we help you today?'", 128, None, 15, "English to Arabic translation", "en", "translation")
p("Translate to Hindi: 'The train arrives at the station at 5:30 PM.'", 128, None, 15, "English to Hindi translation", "en", "translation")
p("Translate to Thai: 'I would like to order pad thai and a glass of water please.'", 128, None, 15, "English to Thai translation", "en", "translation")
p("Translate to Indonesian: 'The deadline for the project is next Monday.'", 128, None, 15, "English to Indonesian translation", "en", "translation")
p("Translate to Portuguese: 'I'm learning to play the guitar in my free time.'", 128, None, 15, "English to Portuguese translation", "en", "translation")
p("Translate to Tagalog: 'Where is the nearest hospital? I need to see a doctor.'", 128, None, 15, "English to Tagalog translation", "en", "translation")

# --- Code generation ---
p("Write a Python function that reverses a string.", 256, None, 30, "Python code generation", "en", "code")
p("Write a Rust function that checks if a number is prime.", 256, None, 30, "Rust code generation", "en", "code")
p("Write a JavaScript function that debounces another function.", 256, None, 30, "JavaScript debounce", "en", "code")
p("Write a SQL query to find the top 5 customers by total purchase amount.", 256, None, 30, "SQL query generation", "en", "code")
p("Write a Python function to merge two sorted lists into one sorted list.", 256, None, 30, "Python merge sorted lists", "en", "code")
p("Write a Rust struct for a 2D point with methods to calculate distance to another point.", 256, None, 30, "Rust struct with methods", "en", "code")

# --- Creative writing ---
p("Write a 4-line haiku about autumn leaves.", 64, None, 15, "Creative writing, short output", "en", "creative")
p("Write a short poem about the ocean in 4 lines.", 64, None, 15, "Short poem", "en", "creative")
p("Write a 3-sentence story about a robot learning to paint.", 128, None, 20, "Flash fiction", "en", "creative")
p("Escribe un haiku de 4 líneas sobre la lluvia.", 64, None, 15, "Spanish haiku", "es", "creative")
p("Write a 3-line haiku about cherry blossoms in Japanese.", 64, None, 15, "Japanese haiku", "ja", "creative")

# --- List generation ---
p("List 5 healthy breakfast ideas with brief descriptions.", 256, None, 40, "List generation", "en", "list")
p("List 5 tips for improving productivity while working from home.", 256, None, 40, "Productivity tips list", "en", "list")
p("Liste 5 idées de déjeuners sains avec de brèves descriptions.", 256, None, 40, "French breakfast list", "fr", "list")
p("Liste 5 Tipps zur Verbesserung der Produktivität beim Arbeiten von zu Hause.", 256, None, 40, "German productivity tips", "de", "list")

# --- Reasoning ---
p("Explain the difference between TCP and UDP in 3-4 sentences.", 256, None, 40, "Technical explanation", "en", "reasoning")
p("Why is encryption important for user privacy? Explain in 3 sentences.", 256, None, 40, "Security explanation", "en", "reasoning")
p("What are the benefits of using containers for deployment? List 3 benefits.", 256, None, 40, "DevOps reasoning", "en", "reasoning")
p("Explique la diferencia entre TCP y UDP en 3-4 oraciones.", 256, None, 40, "Spanish technical explanation", "es", "reasoning")
p("TCPとUDPの違いを3-4文で説明してください。", 256, None, 40, "Japanese technical explanation", "ja", "reasoning")

# --- Instruction following ---
p("Write an email to your manager requesting time off next week. Include: subject line, greeting, reason, dates, and sign-off.", 512, None, 60, "Email writing with structure", "en", "instruction")
p("Write a product description for a smart water bottle. Include: name, features (3), benefits (2), and call to action.", 512, None, 60, "Product description with structure", "en", "instruction")
p("Create a weekly meal plan for a vegetarian diet. Include breakfast, lunch, and dinner for each day (Monday-Friday).", 512, None, 80, "Structured meal plan", "en", "instruction")
p("Write a blog post outline about 'The Future of AI in Healthcare'. Include: title, 5 section headings, and 2 bullet points per section.", 512, None, 80, "Blog post outline", "en", "instruction")

# --- Regex constrained ---
p("Generate a US phone number in the format XXX-XXX-XXXX.", 64,
  {"type": "regex", "pattern": r"\d{3}-\d{3}-\d{4}"},
  12, "Regex constrained: phone number", "en", "regex")

p("Generate an email address for user 'john smith' at company 'example.com'.", 64,
  {"type": "regex", "pattern": r"[a-z]+\.[a-z]+@[a-z]+\.[a-z]+"},
  15, "Regex constrained: email", "en", "regex")

p("Generate a date in YYYY-MM-DD format for March 15, 2026.", 64,
  {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"},
  10, "Regex constrained: date", "en", "regex")

# --- Multilingual instruction following ---
p("次の要件を満たすメールを書いてください：件名、挨拶、来週の休暇申請、署名。", 512, None, 60, "Japanese email writing", "ja", "instruction")
p("Escribe una descripción de producto para una botella de agua inteligente. Incluye: nombre, 3 características, 2 beneficios, y llamada a la acción.", 512, None, 60, "Spanish product description", "es", "instruction")

dataset = {
    "name": "kchat-generation-eval-dataset",
    "version": "2.0.0",
    "description": "Comprehensive generation evaluation dataset with 40+ prompts across 8 categories (greeting, summarization, JSON schema, translation, code, creative, list, reasoning, instruction, regex) and 14 languages.",
    "prompts": prompts,
    "performance_targets": {
        "low_tier": {"ttft_p95_ms": 2500, "decode_p50_tps": 8, "max_memory_gb": 0.75},
        "medium_tier": {"ttft_p95_ms": 1500, "decode_p50_tps": 15, "max_memory_gb": 1.5},
        "high_tier": {"ttft_p95_ms": 1000, "decode_p50_tps": 25, "max_memory_gb": 3.0},
    },
}

with open("generation_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Prompts: {len(prompts)}")
from collections import Counter
cats = Counter(p["category"] for p in prompts)
langs = Counter(p["language"] for p in prompts)
print(f"By category: {dict(cats)}")
print(f"By language: {dict(langs)}")
print(f"File size: {len(json.dumps(dataset))} bytes")
