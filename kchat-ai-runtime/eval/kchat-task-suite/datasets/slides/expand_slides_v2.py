#!/usr/bin/env python3
"""Expand slides dataset to 1100+ cases with multilingual content and language metadata."""
import json, copy

with open("slides_eval_dataset_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["test_cases"]

# 1. Add language field to all existing cases (default "en")
for tc in cases:
    if "language" not in tc:
        tc["language"] = "en"

# 2. Create multilingual variants of existing cases
multilingual_inputs = {
    "ja": {
        "Create a title slide about teamwork": "チームワークについてのタイトルスライドを作成",
        "Create a title slide about innovation": "イノベーションについてのタイトルスライドを作成",
        "Create a title slide about sustainability": "持続可能性についてのタイトルスライドを作成",
        "Create a bullet slide with key points about AI": "AIの主要ポイントをまとめた箇条書きスライドを作成",
        "Create a comparison slide comparing React and Vue": "ReactとVueを比較するスライドを作成",
        "Create a timeline slide showing project milestones": "プロジェクトマイルストーンを示すタイムラインスライドを作成",
        "Create a quote slide with an inspirational message": "インスピレーションを与えるメッセージの引用スライドを作成",
        "Create an agenda slide for a quarterly review": "四半期レビューのアジェンダスライドを作成",
    },
    "ko": {
        "Create a title slide about teamwork": "팀워크에 대한 제목 슬라이드를 만들어주세요",
        "Create a title slide about innovation": "혁신에 대한 제목 슬라이드를 만들어주세요",
        "Create a bullet slide with key points about AI": "AI의 핵심 포인트를 정리한 불릿 슬라이드를 만들어주세요",
        "Create a comparison slide comparing React and Vue": "React와 Vue를 비교하는 슬라이드를 만들어주세요",
        "Create a timeline slide showing project milestones": "프로젝트 마일스톤을 보여주는 타임라인 슬라이드를 만들어주세요",
    },
    "zh": {
        "Create a title slide about teamwork": "创建一个关于团队合作的标题幻灯片",
        "Create a title slide about innovation": "创建一个关于创新的标题幻灯片",
        "Create a bullet slide with key points about AI": "创建一个关于AI要点的项目符号幻灯片",
        "Create a comparison slide comparing React and Vue": "创建一个比较React和Vue的幻灯片",
        "Create a timeline slide showing project milestones": "创建一个显示项目里程碑的时间线幻灯片",
    },
    "es": {
        "Create a title slide about teamwork": "Crea una diapositiva de título sobre trabajo en equipo",
        "Create a title slide about innovation": "Crea una diapositiva de título sobre innovación",
        "Create a bullet slide with key points about AI": "Crea una diapositiva con viñetas sobre puntos clave de IA",
        "Create a comparison slide comparing React and Vue": "Crea una diapositiva comparando React y Vue",
        "Create a timeline slide showing project milestones": "Crea una diapositiva de cronograma con hitos del proyecto",
    },
    "vi": {
        "Create a title slide about teamwork": "Tạo slide tiêu đề về làm việc nhóm",
        "Create a title slide about innovation": "Tạo slide tiêu đề về đổi mới",
        "Create a bullet slide with key points about AI": "Tạo slide gạch đầu dòng về điểm chính của AI",
        "Create a comparison slide comparing React and Vue": "Tạo slide so sánh React và Vue",
        "Create a timeline slide showing project milestones": "Tạo slide dòng thời gian hiển thị mốc dự án",
    },
    "fr": {
        "Create a title slide about teamwork": "Créez une diapositive titre sur le travail d'équipe",
        "Create a title slide about innovation": "Créez une diapositive titre sur l'innovation",
        "Create a bullet slide with key points about AI": "Créez une diapositive à puces sur les points clés de l'IA",
    },
    "de": {
        "Create a title slide about teamwork": "Erstelle eine Titelfolie über Teamarbeit",
        "Create a title slide about innovation": "Erstelle eine Titelfolie über Innovation",
        "Create a bullet slide with key points about AI": "Erstelle eine Stichpunktfolie mit Kernpunkten zu KI",
    },
    "ar": {
        "Create a title slide about teamwork": "أنشئ شريحة عنوان عن العمل الجماعي",
        "Create a title slide about innovation": "أنشئ شريحة عنوان عن الابتكار",
    },
    "hi": {
        "Create a title slide about teamwork": "टीमवर्क के बारे में एक शीर्षक स्लाइड बनाएं",
        "Create a title slide about innovation": "नवाचार के बारे में एक शीर्षक स्लाइड बनाएं",
    },
    "th": {
        "Create a title slide about teamwork": "สร้างสไลด์หัวข้อเรื่องการทำงานเป็นทีม",
        "Create a title slide about innovation": "สร้างสไลด์หัวข้อเรื่องนวัตกรรม",
    },
    "id": {
        "Create a title slide about teamwork": "Buat slide judul tentang kerja tim",
        "Create a title slide about innovation": "Buat slide judul tentang inovasi",
    },
    "pt": {
        "Create a title slide about teamwork": "Crie um slide de título sobre trabalho em equipe",
        "Create a title slide about innovation": "Crie um slide de título sobre inovação",
    },
    "tl": {
        "Create a title slide about teamwork": "Gumawa ng title slide tungkol sa teamwork",
        "Create a title slide about innovation": "Gumawa ng title slide tungkol sa innovation",
    },
}

new_cases = []
max_num = 0
for tc in cases:
    # Extract number from id for counter
    parts = tc["id"].rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        max_num = max(max_num, int(parts[1]))

counter = max_num

# Find template cases (English ones with simple variant_context)
template_cases = {}
for tc in cases:
    vc = tc["input"].get("variant_context", "")
    if vc in multilingual_inputs.get("ja", {}):
        if vc not in template_cases:
            template_cases[vc] = tc

# Create multilingual variants
for lang, translations in multilingual_inputs.items():
    for en_text, translated_text in translations.items():
        if en_text in template_cases:
            base = template_cases[en_text]
            counter += 1
            new_tc = copy.deepcopy(base)
            new_tc["id"] = f"{base['id'].rsplit('_',1)[0]}_{lang}_{counter:04d}"
            new_tc["input"]["variant_context"] = translated_text
            new_tc["language"] = lang
            new_tc["description"] = f"{base['description']} ({lang})"
            new_cases.append(new_tc)

print(f"Original cases: {len(cases)}")
print(f"New multilingual cases: {len(new_cases)}")

cases.extend(new_cases)
print(f"Total cases: {len(cases)}")

# Update metadata
data["version"] = "2.0.0"
data["description"] = "Comprehensive slides AI skill evaluation dataset with 880+ cases across 12 skills and 210 templates, including multilingual cases in 14 languages."

with open("slides_eval_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

from collections import Counter
langs = Counter(tc.get("language", "?") for tc in cases)
skills = Counter(tc.get("skill_id", "?") for tc in cases)
print(f"\nBy language: {dict(langs)}")
print(f"By skill (top 5): {dict(skills.most_common(5))}")
print(f"File size: {len(json.dumps(data))} bytes")
