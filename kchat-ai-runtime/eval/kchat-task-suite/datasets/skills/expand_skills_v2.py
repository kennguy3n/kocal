#!/usr/bin/env python3
"""Expand skills dataset to 300+ cases with multilingual content, translation quality, and tone matching."""
import json, copy

with open("skill_eval_dataset_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["test_cases"]

# 1. Add language field to all existing cases (default "en")
for tc in cases:
    if "language" not in tc:
        tc["language"] = "en"

# 2. Create multilingual variants for translation skills
translation_cases = [tc for tc in cases if "translate" in tc["skill_id"]]
print(f"Translation cases: {len(translation_cases)}")

# Sample documents in different languages for translation tests
multilingual_docs = {
    "ja": "人工知能（AI）は、機械が人間のように学習し推論する能力を指します。機械学習はAIのサブセットで、データからパターンを学びます。深層学習はニューラルネットワークを使用する機械学習の一種です。近年のAIの進歩は主に深層学習によるものです。",
    "ko": "인공지능(AI)은 기계가 인간처럼 학습하고 추론하는 능력을 말합니다. 머신러닝은 AI의 하위 집합으로, 데이터에서 패턴을 학습합니다. 딥러닝은 신경망을 사용하는 머신러닝의 한 종류입니다. 최근 AI의 발전은 주로 딥러닝에 의한 것입니다.",
    "zh": "人工智能（AI）是指机器像人类一样学习和推理的能力。机器学习是AI的子集，从数据中学习模式。深度学习是使用神经网络的机器学习的一种。近年来的AI进步主要由深度学习推动。",
    "es": "La inteligencia artificial (IA) se refiere a la capacidad de las máquinas para aprender y razonar como humanos. El aprendizaje automático es un subconjunto de la IA que aprende patrones de los datos. El aprendizaje profundo es un tipo de aprendizaje automático que usa redes neuronales.",
    "vi": "Trí tuệ nhân tạo (AI) là khả năng máy móc học tập và suy luận như con người. Học máy là tập con của AI, học mẫu từ dữ liệu. Học sâu sử dụng mạng nơ-ron. Tiến bộ AI gần đây chủ yếu nhờ học sâu.",
    "fr": "L'intelligence artificielle (IA) désigne la capacité des machines à apprendre et à raisonner comme les humains. L'apprentissage automatique est un sous-ensemble de l'IA qui apprend des modèles à partir des données. L'apprentissage profond est un type d'apprentissage automatique utilisant des réseaux de neurones.",
    "de": "Künstliche Intelligenz (KI) bezeichnet die Fähigkeit von Maschinen, wie Menschen zu lernen und zu schlussfolgern. Maschinelles Lernen ist eine Teilmenge der KI, die Muster aus Daten lernt. Deep Learning ist eine Art des maschinellen Lernens, die neuronale Netze verwendet.",
    "ar": "يشير الذكاء الاصطناعي (AI) إلى قدرة الآلات على التعلم والاستدلال مثل البشر. التعلم الآلي هو مجموعة فرعية من الذكاء الاصطناعي يتعلم الأنماط من البيانات. التعلم العميق هو نوع من التعلم الآلي يستخدم الشبكات العصبية.",
    "hi": "कृत्रिम बुद्धिमत्ता (AI) मशीनों की मनुष्यों की तरह सीखने और तर्क करने की क्षमता को संदर्भित करता है। मशीन लर्निंग AI का एक उपसमुच्छय है जो डेटा से पैटर्न सीखता है। डीप लर्निंग न्यूरल नेटवर्क का उपयोग करने वाला मशीन लर्निंग का एक प्रकार है।",
    "th": "ปัญญาประดิษฐ์ (AI) หมายถึงความสามารถของเครื่องจักรในการเรียนรู้และใช้เหตุผลเหมือนมนุษย์ การเรียนรู้ของเครื่องเป็นส่วนย่อยของ AI ที่เรียนรู้รูปแบบจากข้อมูล การเรียนรู้เชิงลึกเป็นประเภทหนึ่งของการเรียนรู้ของเครื่องที่ใช้โครงข่ายประสาทเทียม",
    "id": "Kecerdasan buatan (AI) mengacu pada kemampuan mesin untuk belajar dan menalar seperti manusia. Pembelajaran mesin adalah subset dari AI yang belajar pola dari data. Pembelajaran mendalam adalah jenis pembelajaran mesin yang menggunakan jaringan saraf.",
    "pt": "A inteligência artificial (IA) refere-se à capacidade das máquinas de aprender e raciocinar como humanos. O aprendizado de máquina é um subconjunto da IA que aprende padrões dos dados. O aprendizado profundo é um tipo de aprendizado de máquina que usa redes neurais.",
    "tl": "Ang artificial intelligence (AI) ay tumutukoy sa kakayahan ng mga makina na mag-aral at mag-reason tulad ng mga tao. Ang machine learning ay isang subset ng AI na nag-aaral ng mga pattern mula sa data. Ang deep learning ay isang uri ng machine learning na gumagamit ng neural networks.",
}

# Tone matching cases
tone_variants = ["professional", "casual", "friendly", "formal", "persuasive", "apologetic", "enthusiastic", "empathetic"]

# Create multilingual translation cases
new_cases = []
max_num = 0
for tc in cases:
    # Extract number from id
    parts = tc["id"].rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        max_num = max(max_num, int(parts[1]))

counter = max_num

# Add multilingual translation cases
for lang, doc_text in multilingual_docs.items():
    for base in translation_cases[:2]:  # 2 base cases per language
        counter += 1
        new_tc = copy.deepcopy(base)
        new_tc["id"] = f"skill_{base['skill_id']}_{lang}_{counter:04d}"
        new_tc["input"]["document"] = doc_text
        new_tc["input"]["selection"] = doc_text[:200]
        new_tc["language"] = lang
        new_tc["description"] = f"{base['description']} (source: {lang})"
        new_cases.append(new_tc)

print(f"Multilingual translation cases: {len(new_cases)}")

# Add tone matching cases
tone_base = next(tc for tc in cases if tc["skill_id"] == "edit_change_tone")
sample_doc = "Hey team, just wanted to let you know that the project is delayed. We need more time to fix the bugs. Sorry about that."

for tone in tone_variants:
    counter += 1
    new_tc = copy.deepcopy(tone_base)
    new_tc["id"] = f"skill_edit_change_tone_{tone}_{counter:04d}"
    new_tc["input"]["document"] = sample_doc
    new_tc["input"]["selection"] = sample_doc
    new_tc["input"]["variant_context"] = f"Change the tone to {tone}"
    new_tc["variant"] = tone
    new_tc["language"] = "en"
    new_tc["description"] = f"Change tone to {tone}"
    # Add tone-specific quality check
    new_tc["quality_checks"].append({
        "type": "contains_keyword",
        "keywords": {
            "professional": ["regret", "inform", "update"],
            "casual": ["hey", "heads up", "just so you know"],
            "friendly": ["hi", "wanted", "thanks"],
            "formal": ["regret", "inform", "aforementioned"],
            "persuasive": ["should", "recommend", "opportunity"],
            "apologetic": ["apologize", "sorry", "regret"],
            "enthusiastic": ["excited", "great", "looking forward"],
            "empathetic": ["understand", "appreciate", "recognize"],
        }.get(tone, [])
    })
    new_cases.append(new_tc)

print(f"Tone matching cases: {len(tone_variants)}")

# Add multilingual summarization cases
summarize_base = next(tc for tc in cases if tc["skill_id"] == "doc_summarize")
for lang, doc_text in multilingual_docs.items():
    counter += 1
    new_tc = copy.deepcopy(summarize_base)
    new_tc["id"] = f"skill_doc_summarize_{lang}_{counter:04d}"
    new_tc["input"]["document"] = doc_text
    new_tc["language"] = lang
    new_tc["description"] = f"Summarize {lang} document"
    new_cases.append(new_tc)

print(f"Multilingual summarization cases: {len(multilingual_docs)}")

# Add multilingual key points extraction
keypoints_base = next(tc for tc in cases if tc["skill_id"] == "doc_key_points")
for lang, doc_text in list(multilingual_docs.items())[:7]:
    counter += 1
    new_tc = copy.deepcopy(keypoints_base)
    new_tc["id"] = f"skill_doc_key_points_{lang}_{counter:04d}"
    new_tc["input"]["document"] = doc_text
    new_tc["language"] = lang
    new_tc["description"] = f"Extract key points from {lang} document"
    new_cases.append(new_tc)

print(f"Multilingual key points cases: 7")

# Add multilingual grammar fix cases
grammar_base = next(tc for tc in cases if tc["skill_id"] == "edit_fix_grammar")
grammar_docs = {
    "ja": "昨日、私は友達と映画を見に行きました。映画はとても面白かったです。私たちはポップコーンを食べながら楽しく過ごしました。",
    "es": "Ayer, yo fui al cine con mi amigo. La película fue muy interesante. Nosotros comimos palomitas mientras disfrutamos.",
    "fr": "Hier, je suis allé au cinéma avec mon ami. Le film était très intéressant. Nous avons mangé du popcorn en profitant.",
}
for lang, doc_text in grammar_docs.items():
    counter += 1
    new_tc = copy.deepcopy(grammar_base)
    new_tc["id"] = f"skill_edit_fix_grammar_{lang}_{counter:04d}"
    new_tc["input"]["document"] = doc_text
    new_tc["input"]["selection"] = doc_text
    new_tc["language"] = lang
    new_tc["description"] = f"Fix grammar in {lang} text"
    new_cases.append(new_tc)

print(f"Multilingual grammar fix cases: {len(grammar_docs)}")

# Add multilingual brainstorm cases
brainstorm_base = next(tc for tc in cases if tc["skill_id"] == "create_brainstorm")
brainstorm_topics = {
    "ja": "モバイルアプリのプライバシー機能のアイデア",
    "ko": "모바일 앱 프라이버시 기능 아이디어",
    "zh": "移动应用隐私功能的创意",
    "es": "Ideas para funciones de privacidad en aplicaciones móviles",
    "vi": "Ý tưởng tính năng quyền riêng tư cho ứng dụng di động",
    "fr": "Idées de fonctionnalités de confidentialité pour applications mobiles",
    "de": "Ideen für Datenschutz-Funktionen in mobilen Apps",
}
for lang, topic in brainstorm_topics.items():
    counter += 1
    new_tc = copy.deepcopy(brainstorm_base)
    new_tc["id"] = f"skill_create_brainstorm_{lang}_{counter:04d}"
    new_tc["input"]["variant_context"] = topic
    new_tc["language"] = lang
    new_tc["description"] = f"Brainstorm in {lang}: {topic}"
    new_cases.append(new_tc)

print(f"Multilingual brainstorm cases: {len(brainstorm_topics)}")

# Add multilingual email draft cases
email_base = next(tc for tc in cases if tc["skill_id"] == "create_email_draft")
email_contexts = {
    "ja": "チームメンバーへのプロジェクト遅延の通知メール",
    "ko": "팀원에게 프로젝트 지연을 알리는 이메일",
    "zh": "通知团队成员项目延迟的邮件",
    "es": "Email al equipo notificando retraso del proyecto",
    "vi": "Email thông báo cho đội về việc dự án bị trễ",
    "fr": "Email à l'équipe notifiant un retard de projet",
    "de": "E-Mail an das Team über Projektverzögerung",
}
for lang, ctx in email_contexts.items():
    counter += 1
    new_tc = copy.deepcopy(email_base)
    new_tc["id"] = f"skill_create_email_draft_{lang}_{counter:04d}"
    new_tc["input"]["variant_context"] = ctx
    new_tc["language"] = lang
    new_tc["description"] = f"Email draft in {lang}"
    new_cases.append(new_tc)

print(f"Multilingual email draft cases: {len(email_contexts)}")

# Add multilingual SEO meta cases
seo_base = next(tc for tc in cases if tc["skill_id"] == "create_seo_meta")
seo_docs = {
    "es": "Guía completa de privacidad en aplicaciones móviles. Aprende cómo proteger tus datos personales, configurar permisos y usar VPN para mayor seguridad online.",
    "fr": "Guide complet de confidentialité des applications mobiles. Apprenez à protéger vos données personnelles, configurer les autorisations et utiliser un VPN.",
    "de": "Vollständiger Leitfaden zum Datenschutz in mobilen Apps. Erfahren Sie, wie Sie persönliche Daten schützen, Berechtigungen konfigurieren und VPN nutzen.",
}
for lang, doc_text in seo_docs.items():
    counter += 1
    new_tc = copy.deepcopy(seo_base)
    new_tc["id"] = f"skill_create_seo_meta_{lang}_{counter:04d}"
    new_tc["input"]["document"] = doc_text
    new_tc["language"] = lang
    new_tc["description"] = f"SEO meta in {lang}"
    new_cases.append(new_tc)

print(f"Multilingual SEO meta cases: {len(seo_docs)}")

# Add multilingual title suggestion cases
title_base = next(tc for tc in cases if tc["skill_id"] == "create_suggest_title")
title_docs = {
    "ja": "オンデバイスAIの未来：プライバシーを守る次世代のモバイルアプリケーション",
    "ko": "온디바이스 AI의 미래: 프라이버시를 보호하는 차세대 모바일 애플리케이션",
    "zh": "端侧AI的未来：保护隐私的下一代移动应用",
    "es": "El futuro de la IA en el dispositivo: aplicaciones móviles de próxima generación que protegen la privacidad",
    "vi": "Tương lai của AI trên thiết bị: ứng dụng di động thế hệ mới bảo vệ quyền riêng tư",
}
for lang, doc_text in title_docs.items():
    counter += 1
    new_tc = copy.deepcopy(title_base)
    new_tc["id"] = f"skill_create_suggest_title_{lang}_{counter:04d}"
    new_tc["input"]["document"] = doc_text
    new_tc["language"] = lang
    new_tc["description"] = f"Suggest title in {lang}"
    new_cases.append(new_tc)

print(f"Multilingual title suggestion cases: {len(title_docs)}")

cases.extend(new_cases)
print(f"\nTotal cases: {len(cases)}")

data["version"] = "2.0.0"
data["description"] = f"Comprehensive skills evaluation dataset with {len(cases)} cases across 33 skills, including multilingual translation, tone matching, and cross-language quality checks."

with open("skill_eval_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

from collections import Counter
langs = Counter(tc.get("language", "?") for tc in cases)
skills = Counter(tc.get("skill_id", "?") for tc in cases)
print(f"\nBy language: {dict(langs)}")
print(f"Total skills: {len(skills)}")
print(f"File size: {len(json.dumps(data))} bytes")
