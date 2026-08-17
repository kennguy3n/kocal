#!/usr/bin/env python3
"""Add 200+ multilingual slides cases by creating translated variants across 14 languages."""
import json, copy, re

with open("slides_eval_dataset_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["test_cases"]

# Topic translations for creating multilingual slide generation prompts
topics = [
    "teamwork", "innovation", "sustainability", "artificial intelligence",
    "cloud computing", "data privacy", "customer success", "product launch",
    "market analysis", "quarterly review", "project milestones", "team building",
]

templates = [
    "Create a title slide about {topic}",
    "Create a bullet slide with key points about {topic}",
    "Create a comparison slide about {topic}",
    "Create a timeline slide showing {topic}",
    "Create an agenda slide for a {topic} meeting",
    "Create a quote slide about {topic}",
    "Summarize key takeaways about {topic}",
    "Create a section divider slide for {topic}",
]

translations = {
    "ja": {"teamwork":"チームワーク","innovation":"イノベーション","sustainability":"持続可能性","artificial intelligence":"人工知能","cloud computing":"クラウドコンピューティング","data privacy":"データプライバシー","customer success":"顧客成功","product launch":"製品ローンチ","market analysis":"市場分析","quarterly review":"四半期レビュー","project milestones":"プロジェクトマイルストーン","team building":"チームビルディング","Create a title slide about":"タイトルスライドを作成:","Create a bullet slide with key points about":"箇条書きスライドを作成:","Create a comparison slide about":"比較スライドを作成:","Create a timeline slide showing":"タイムラインスライドを作成:","Create an agenda slide for a":"アジェンダスライドを作成:","meeting":"会議","Create a quote slide about":"引用スライドを作成:","Summarize key takeaways about":"要点をまとめる:","Create a section divider slide for":"セクション分割スライドを作成:"},
    "ko": {"teamwork":"팀워크","innovation":"혁신","sustainability":"지속가능성","artificial intelligence":"인공지능","cloud computing":"클라우드 컴퓨팅","data privacy":"데이터 프라이버시","customer success":"고객 성공","product launch":"제품 출시","market analysis":"시장 분석","quarterly review":"분기 리뷰","project milestones":"프로젝트 마일스톤","team building":"팀 빌딩","Create a title slide about":"제목 슬라이드 만들기:","Create a bullet slide with key points about":"핵심 포인트 불릿 슬라이드 만들기:","Create a comparison slide about":"비교 슬라이드 만들기:","Create a timeline slide showing":"타임라인 슬라이드 만들기:","Create an agenda slide for a":"아젠다 슬라이드 만들기:","meeting":"회의","Create a quote slide about":"인용 슬라이드 만들기:","Summarize key takeaways about":"핵심 요약:","Create a section divider slide for":"섹션 구분 슬라이드 만들기:"},
    "zh": {"teamwork":"团队合作","innovation":"创新","sustainability":"可持续发展","artificial intelligence":"人工智能","cloud computing":"云计算","data privacy":"数据隐私","customer success":"客户成功","product launch":"产品发布","market analysis":"市场分析","quarterly review":"季度回顾","project milestones":"项目里程碑","team building":"团队建设","Create a title slide about":"创建标题幻灯片:","Create a bullet slide with key points about":"创建要点项目符号幻灯片:","Create a comparison slide about":"创建比较幻灯片:","Create a timeline slide showing":"创建时间线幻灯片:","Create an agenda slide for a":"创建议程幻灯片:","meeting":"会议","Create a quote slide about":"创建引用幻灯片:","Summarize key takeaways about":"总结要点:","Create a section divider slide for":"创建章节分割幻灯片:"},
    "es": {"teamwork":"trabajo en equipo","innovation":"innovación","sustainability":"sostenibilidad","artificial intelligence":"inteligencia artificial","cloud computing":"computación en la nube","data privacy":"privacidad de datos","customer success":"éxito del cliente","product launch":"lanzamiento de producto","market analysis":"análisis de mercado","quarterly review":"revisión trimestral","project milestones":"hitos del proyecto","team building":"construcción de equipo","Create a title slide about":"Crea una diapositiva de título sobre","Create a bullet slide with key points about":"Crea una diapositiva con viñetas sobre","Create a comparison slide about":"Crea una diapositiva de comparación sobre","Create a timeline slide showing":"Crea una diapositiva de cronograma mostrando","Create an agenda slide for a":"Crea una diapositiva de agenda para una reunión de","meeting":"reunión","Create a quote slide about":"Crea una diapositiva de cita sobre","Summarize key takeaways about":"Resume los puntos clave sobre","Create a section divider slide for":"Crea una diapositiva divisoria de sección para"},
    "vi": {"teamwork":"làm việc nhóm","innovation":"đổi mới","sustainability":"phát triển bền vững","artificial intelligence":"trí tuệ nhân tạo","cloud computing":"điện toán đám mây","data privacy":"quyền riêng tư dữ liệu","customer success":"thành công khách hàng","product launch":"ra mắt sản phẩm","market analysis":"phân tích thị trường","quarterly review":"đánh giá hàng quý","project milestones":"mốc dự án","team building":"xây dựng đội ngũ","Create a title slide about":"Tạo slide tiêu đề về","Create a bullet slide with key points about":"Tạo slide gạch đầu dòng về","Create a comparison slide about":"Tạo slide so sánh về","Create a timeline slide showing":"Tạo slide dòng thời gian hiển thị","Create an agenda slide for a":"Tạo slide chương trình nghị sự cho cuộc họp","meeting":"","Create a quote slide about":"Tạo slide trích dẫn về","Summarize key takeaways about":"Tóm tắt điểm chính về","Create a section divider slide for":"Tạo slide phân chia phần cho"},
    "fr": {"teamwork":"travail d'équipe","innovation":"innovation","sustainability":"durabilité","artificial intelligence":"intelligence artificielle","cloud computing":"informatique en nuage","data privacy":"confidentialité des données","customer success":"succès client","product launch":"lancement produit","market analysis":"analyse de marché","quarterly review":"revue trimestrielle","project milestones":"jalons du projet","team building":"cohésion d'équipe","Create a title slide about":"Créez une diapositive titre sur","Create a bullet slide with key points about":"Créez une diapositive à puces sur","Create a comparison slide about":"Créez une diapositive de comparaison sur","Create a timeline slide showing":"Créez une diapositive chronologique montrant","Create an agenda slide for a":"Créez une diapositive d'agenda pour une réunion","meeting":"réunion","Create a quote slide about":"Créez une diapositive de citation sur","Summarize key takeaways about":"Résumez les points clés sur","Create a section divider slide for":"Créez une diapositive de séparation pour"},
    "de": {"teamwork":"Teamarbeit","innovation":"Innovation","sustainability":"Nachhaltigkeit","artificial intelligence":"künstliche Intelligenz","cloud computing":"Cloud-Computing","data privacy":"Datenschutz","customer success":"Kundenerfolg","product launch":"Produktlaunch","market analysis":"Marktanalyse","quarterly review":"Quartalsreview","project milestones":"Projektmeilensteine","team building":"Teambuilding","Create a title slide about":"Erstelle eine Titelfolie über","Create a bullet slide with key points about":"Erstelle eine Stichpunktfolie über","Create a comparison slide about":"Erstelle eine Vergleichsfolie über","Create a timeline slide showing":"Erstelle eine Zeitstrahlfolie über","Create an agenda slide for a":"Erstelle eine Agenda-Folie für ein Meeting über","meeting":"","Create a quote slide about":"Erstelle eine Zitatfolie über","Summarize key takeaways about":"Fasse die Kernpunkte zusammen über","Create a section divider slide for":"Erstelle eine Trennfolie für"},
    "ar": {"teamwork":"العمل الجماعي","innovation":"الابتكار","sustainability":"الاستدامة","artificial intelligence":"الذكاء الاصطناعي","cloud computing":"الحوسبة السحابية","data privacy":"خصوصية البيانات","customer success":"نجاح العملاء","product launch":"إطلاق المنتج","market analysis":"تحليل السوق","quarterly review":"مراجعة ربع سنوية","project milestones":"معالم المشروع","team building":"بناء الفريق","Create a title slide about":"أنشئ شريحة عنوان عن","Create a bullet slide with key points about":"أنشئ شريحة نقاط رئيسية عن","Create a comparison slide about":"أنشئ شريحة مقارنة عن","Create a timeline slide showing":"أنشئ شريحة خط زمني تظهر","Create an agenda slide for a":"أنشئ شريحة جدول أعمال لاجتماع","meeting":"","Create a quote slide about":"أنشئ شريحة اقتباس عن","Summarize key takeaways about":"لخص النقاط الرئيسية عن","Create a section divider slide for":"أنشئ شريحة فاصل قسم ل"},
    "hi": {"teamwork":"टीमवर्क","innovation":"नवाचार","sustainability":"स्थिरता","artificial intelligence":"कृत्रिम बुद्धिमत्ता","cloud computing":"क्लाउड कंप्यूटिंग","data privacy":"डेटा गोपनीयता","customer success":"ग्राहक सफलता","product launch":"उत्पाद लॉन्च","market analysis":"बाजार विश्लेषण","quarterly review":"तिमाही समीक्षा","project milestones":"परियोजना मील के पत्थर","team building":"टीम बिल्डिंग","Create a title slide about":"शीर्षक स्लाइड बनाएं:","Create a bullet slide with key points about":"मुख्य बिंदु स्लाइड बनाएं:","Create a comparison slide about":"तुलना स्लाइड बनाएं:","Create a timeline slide showing":"समयरेखा स्लाइड बनाएं:","Create an agenda slide for a":"कार्यसूची स्लाइड बनाएं:","meeting":"बैठक","Create a quote slide about":"उद्धरण स्लाइड बनाएं:","Summarize key takeaways about":"मुख्य निष्कर्ष:","Create a section divider slide for":"अनुभाग विभाजक स्लाइड बनाएं:"},
    "th": {"teamwork":"การทำงานเป็นทีม","innovation":"นวัตกรรม","sustainability":"ความยั่งยืน","artificial intelligence":"ปัญญาประดิษฐ์","cloud computing":"คลาวด์คอมพิวติ้ง","data privacy":"ความเป็นส่วนตัวของข้อมูล","customer success":"ความสำเร็จของลูกค้า","product launch":"เปิดตัวผลิตภัณฑ์","market analysis":"วิเคราะห์ตลาด","quarterly review":"รีวิวไตรมาส","project milestones":"เป้าหมายโครงการ","team building":"ทีมบิลดิ้ง","Create a title slide about":"สร้างสไลด์หัวข้อเรื่อง","Create a bullet slide with key points about":"สร้างสไลด์หัวข้อเรื่อง","Create a comparison slide about":"สร้างสไลด์เปรียบเทียบเรื่อง","Create a timeline slide showing":"สร้างสไลด์ไทม์ไลน์แสดง","Create an agenda slide for a":"สร้างสไลด์วาระการประชุม","meeting":"การประชุม","Create a quote slide about":"สร้างสไลด์คำคมเรื่อง","Summarize key takeaways about":"สรุปประเด็นสำคัญเรื่อง","Create a section divider slide for":"สร้างสไลด์แบ่งส่วนสำหรับ"},
    "id": {"teamwork":"kerja tim","innovation":"inovasi","sustainability":"keberlanjutan","artificial intelligence":"kecerdasan buatan","cloud computing":"komputasi awan","data privacy":"privasi data","customer success":"keberhasilan pelanggan","product launch":"peluncuran produk","market analysis":"analisis pasar","quarterly review":"tinjauan kuartal","project milestones":"tonggak proyek","team building":"team building","Create a title slide about":"Buat slide judul tentang","Create a bullet slide with key points about":"Buat slide poin utama tentang","Create a comparison slide about":"Buat slide perbandingan tentang","Create a timeline slide showing":"Buat slide linimasa menampilkan","Create an agenda slide for a":"Buat slide agenda untuk rapat","meeting":"","Create a quote slide about":"Buat slide kutipan tentang","Summarize key takeaways about":"Ringkas poin utama tentang","Create a section divider slide for":"Buat slide pembatas bagian untuk"},
    "pt": {"teamwork":"trabalho em equipe","innovation":"inovação","sustainability":"sustentabilidade","artificial intelligence":"inteligência artificial","cloud computing":"computação em nuvem","data privacy":"privacidade de dados","customer success":"sucesso do cliente","product launch":"lançamento de produto","market analysis":"análise de mercado","quarterly review":"revisão trimestral","project milestones":"marcos do projeto","team building":"construção de equipe","Create a title slide about":"Crie um slide de título sobre","Create a bullet slide with key points about":"Crie um slide com pontos principais sobre","Create a comparison slide about":"Crie um slide de comparação sobre","Create a timeline slide showing":"Crie um slide de linha do tempo mostrando","Create an agenda slide for a":"Crie um slide de agenda para uma reunião","meeting":"reunião","Create a quote slide about":"Crie um slide de citação sobre","Summarize key takeaways about":"Resuma os pontos principais sobre","Create a section divider slide for":"Crie um slide divisor de seção para"},
    "tl": {"teamwork":"teamwork","innovation":"innovation","sustainability":"sustainability","artificial intelligence":"artificial intelligence","cloud computing":"cloud computing","data privacy":"data privacy","customer success":"customer success","product launch":"product launch","market analysis":"market analysis","quarterly review":"quarterly review","project milestones":"project milestones","team building":"team building","Create a title slide about":"Gumawa ng title slide tungkol sa","Create a bullet slide with key points about":"Gumawa ng bullet slide tungkol sa","Create a comparison slide about":"Gumawa ng comparison slide tungkol sa","Create a timeline slide showing":"Gumawa ng timeline slide na nagpapakita ng","Create an agenda slide for a":"Gumawa ng agenda slide para sa","meeting":"meeting","Create a quote slide about":"Gumawa ng quote slide tungkol sa","Summarize key takeaways about":"Ibuod ang mga pangunahing punto tungkol sa","Create a section divider slide for":"Gumawa ng section divider slide para sa"},
}

# Find a base template case (title slide) to use as template
base_cases = {}
for tc in cases:
    if tc["skill_id"] == "slides_generate_slide" and tc.get("language") == "en":
        vc = tc["input"].get("variant_context", "")
        if "Create a title slide about teamwork" in vc:
            base_cases["title"] = tc
            break

# Get the first generate_slide case as a generic template
generic_base = next(tc for tc in cases if tc["skill_id"] == "slides_generate_slide" and tc.get("language") == "en")

max_num = 0
for tc in cases:
    parts = tc["id"].rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        max_num = max(max_num, int(parts[1]))

counter = max_num
new_cases = []

for lang, trans_map in translations.items():
    for template in templates:
        for topic in topics:
            # Translate the template
            translated = template
            # First translate the action phrase
            for en_phrase, translated_phrase in trans_map.items():
                if en_phrase in translated and not en_phrase.islower():
                    translated = translated.replace(en_phrase, translated_phrase)
            # Then translate the topic
            if topic in trans_map:
                translated = translated.replace(topic, trans_map[topic])
            
            # Skip if no translation happened (for tl which is mostly English)
            if translated == template and lang != "tl":
                continue
            
            counter += 1
            new_tc = copy.deepcopy(generic_base)
            new_tc["id"] = f"slide_{lang}_{counter:04d}"
            new_tc["input"]["variant_context"] = translated
            new_tc["language"] = lang
            new_tc["description"] = f"Multilingual slide generation ({lang}): {topic}"
            new_cases.append(new_tc)

print(f"New multilingual cases: {len(new_cases)}")
cases.extend(new_cases)
print(f"Total cases: {len(cases)}")

data["version"] = "2.2.0"
data["description"] = f"Comprehensive slides AI skill evaluation dataset with {len(cases)} cases across 12 skills and 210 templates, including multilingual cases in 14 languages."

with open("slides_eval_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

from collections import Counter
langs = Counter(tc.get("language", "?") for tc in cases)
print(f"\nBy language: {dict(langs)}")
print(f"File size: {len(json.dumps(data))} bytes")
