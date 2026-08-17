#!/usr/bin/env python3
"""Add more multilingual slides cases by translating variant_contexts."""
import json, copy

with open("slides_eval_dataset_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["test_cases"]

# Translation map for common slide topics
topic_translations = {
    "ja": {
        "teamwork": "チームワーク", "innovation": "イノベーション", "sustainability": "持続可能性",
        "AI": "AI", "React and Vue": "ReactとVue", "project milestones": "プロジェクトマイルストーン",
        "quarterly review": "四半期レビュー", "customer success": "顧客成功",
        "data privacy": "データプライバシー", "cloud migration": "クラウド移行",
        "product launch": "製品ローンチ", "market analysis": "市場分析",
        "team building": "チームビルディング", "OKR": "OKR",
    },
    "ko": {
        "teamwork": "팀워크", "innovation": "혁신", "sustainability": "지속가능성",
        "AI": "AI", "React and Vue": "React와 Vue", "project milestones": "프로젝트 마일스톤",
        "quarterly review": "분기 리뷰", "customer success": "고객 성공",
        "data privacy": "데이터 프라이버시", "cloud migration": "클라우드 마이그레이션",
        "product launch": "제품 출시", "market analysis": "시장 분석",
        "team building": "팀 빌딩", "OKR": "OKR",
    },
    "zh": {
        "teamwork": "团队合作", "innovation": "创新", "sustainability": "可持续发展",
        "AI": "AI", "React and Vue": "React和Vue", "project milestones": "项目里程碑",
        "quarterly review": "季度回顾", "customer success": "客户成功",
        "data privacy": "数据隐私", "cloud migration": "云迁移",
        "product launch": "产品发布", "market analysis": "市场分析",
        "team building": "团队建设", "OKR": "OKR",
    },
    "es": {
        "teamwork": "trabajo en equipo", "innovation": "innovación", "sustainability": "sostenibilidad",
        "AI": "IA", "React and Vue": "React y Vue", "project milestones": "hitos del proyecto",
        "quarterly review": "revisión trimestral", "customer success": "éxito del cliente",
        "data privacy": "privacidad de datos", "cloud migration": "migración a la nube",
        "product launch": "lanzamiento de producto", "market analysis": "análisis de mercado",
        "team building": "construcción de equipo", "OKR": "OKR",
    },
    "vi": {
        "teamwork": "làm việc nhóm", "innovation": "đổi mới", "sustainability": "phát triển bền vững",
        "AI": "AI", "React and Vue": "React và Vue", "project milestones": "mốc dự án",
        "quarterly review": "đánh giá hàng quý", "customer success": "thành công khách hàng",
        "data privacy": "quyền riêng tư dữ liệu", "cloud migration": "di chuyển đám mây",
        "product launch": "ra mắt sản phẩm", "market analysis": "phân tích thị trường",
        "team building": "xây dựng đội ngũ", "OKR": "OKR",
    },
    "fr": {
        "teamwork": "travail d'équipe", "innovation": "innovation", "sustainability": "durabilité",
        "AI": "IA", "React and Vue": "React et Vue", "project milestones": "jalons du projet",
        "quarterly review": "revue trimestrielle", "customer success": "succès client",
        "data privacy": "confidentialité des données", "cloud migration": "migration cloud",
        "product launch": "lancement produit", "market analysis": "analyse de marché",
        "team building": "cohésion d'équipe", "OKR": "OKR",
    },
    "de": {
        "teamwork": "Teamarbeit", "innovation": "Innovation", "sustainability": "Nachhaltigkeit",
        "AI": "KI", "React and Vue": "React und Vue", "project milestones": "Projektmeilensteine",
        "quarterly review": "Quartalsreview", "customer success": "Kundenerfolg",
        "data privacy": "Datenschutz", "cloud migration": "Cloud-Migration",
        "product launch": "Produktlaunch", "market analysis": "Marktanalyse",
        "team building": "Teambuilding", "OKR": "OKR",
    },
    "ar": {
        "teamwork": "العمل الجماعي", "innovation": "الابتكار", "sustainability": "الاستدامة",
        "AI": "الذكاء الاصطناعي", "React and Vue": "React و Vue", "project milestones": "معالم المشروع",
        "quarterly review": "مراجعة ربع سنوية", "customer success": "نجاح العملاء",
        "data privacy": "خصوصية البيانات", "cloud migration": "الانتقال إلى السحابة",
        "product launch": "إطلاق المنتج", "market analysis": "تحليل السوق",
        "team building": "بناء الفريق", "OKR": "OKR",
    },
    "hi": {
        "teamwork": "टीमवर्क", "innovation": "नवाचार", "sustainability": "स्थिरता",
        "AI": "एआई", "React and Vue": "React और Vue", "project milestones": "परियोजना मील के पत्थर",
        "quarterly review": "तिमाही समीक्षा", "customer success": "ग्राहक सफलता",
        "data privacy": "डेटा गोपनीयता", "cloud migration": "क्लाउड माइग्रेशन",
        "product launch": "उत्पाद लॉन्च", "market analysis": "बाजार विश्लेषण",
        "team building": "टीम बिल्डिंग", "OKR": "OKR",
    },
    "th": {
        "teamwork": "การทำงานเป็นทีม", "innovation": "นวัตกรรม", "sustainability": "ความยั่งยืน",
        "AI": "AI", "React and Vue": "React และ Vue", "project milestones": "เป้าหมายโครงการ",
        "quarterly review": "รีวิวไตรมาส", "customer success": "ความสำเร็จของลูกค้า",
        "data privacy": "ความเป็นส่วนตัวของข้อมูล", "cloud migration": "ย้ายไปคลาวด์",
        "product launch": "เปิดตัวผลิตภัณฑ์", "market analysis": "วิเคราะห์ตลาด",
        "team building": "ทีมบิลดิ้ง", "OKR": "OKR",
    },
    "id": {
        "teamwork": "kerja tim", "innovation": "inovasi", "sustainability": "keberlanjutan",
        "AI": "AI", "React and Vue": "React dan Vue", "project milestones": "tonggak proyek",
        "quarterly review": "tinjauan kuartal", "customer success": "keberhasilan pelanggan",
        "data privacy": "privasi data", "cloud migration": "migrasi cloud",
        "product launch": "peluncuran produk", "market analysis": "analisis pasar",
        "team building": "team building", "OKR": "OKR",
    },
    "pt": {
        "teamwork": "trabalho em equipe", "innovation": "inovação", "sustainability": "sustentabilidade",
        "AI": "IA", "React and Vue": "React e Vue", "project milestones": "marcos do projeto",
        "quarterly review": "revisão trimestral", "customer success": "sucesso do cliente",
        "data privacy": "privacidade de dados", "cloud migration": "migração para nuvem",
        "product launch": "lançamento de produto", "market analysis": "análise de mercado",
        "team building": "construção de equipe", "OKR": "OKR",
    },
    "tl": {
        "teamwork": "teamwork", "innovation": "innovation", "sustainability": "sustainability",
        "AI": "AI", "React and Vue": "React at Vue", "project milestones": "project milestones",
        "quarterly review": "quarterly review", "customer success": "customer success",
        "data privacy": "data privacy", "cloud migration": "cloud migration",
        "product launch": "product launch", "market analysis": "market analysis",
        "team building": "team building", "OKR": "OKR",
    },
}

# Find generate_slide cases with simple variant_contexts and create multilingual variants
generate_slide_cases = [tc for tc in cases if tc["skill_id"] == "slides_generate_slide" and tc.get("language") == "en"]

# Pick a diverse subset (every 7th case) to create multilingual variants
selected = generate_slide_cases[::7]  # ~120 cases
print(f"Selected {len(selected)} base cases for multilingual expansion")

new_cases = []
max_num = 0
for tc in cases:
    parts = tc["id"].rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        max_num = max(max_num, int(parts[1]))

counter = max_num

for base in selected:
    vc = base["input"].get("variant_context", "")
    # Try to find a translation for this variant_context
    for lang, trans_map in topic_translations.items():
        # Check if any topic keyword is in the variant_context
        translated_vc = vc
        for en_word, translated in trans_map.items():
            if en_word in translated_vc.lower():
                translated_vc = translated_vc.replace(en_word, translated)
        
        # Only add if translation actually changed something
        if translated_vc != vc:
            counter += 1
            new_tc = copy.deepcopy(base)
            new_tc["id"] = f"slide_{lang}_{counter:04d}"
            new_tc["input"]["variant_context"] = translated_vc
            new_tc["language"] = lang
            new_tc["description"] = f"{base['description']} ({lang})"
            new_cases.append(new_tc)
            break  # One language per base case to avoid too many

print(f"New multilingual cases: {len(new_cases)}")
cases.extend(new_cases)
print(f"Total cases: {len(cases)}")

data["version"] = "2.1.0"
data["description"] = f"Comprehensive slides AI skill evaluation dataset with {len(cases)} cases across 12 skills and 210 templates, including multilingual cases in 14 languages."

with open("slides_eval_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

from collections import Counter
langs = Counter(tc.get("language", "?") for tc in cases)
print(f"\nBy language: {dict(langs)}")
print(f"File size: {len(json.dumps(data))} bytes")
