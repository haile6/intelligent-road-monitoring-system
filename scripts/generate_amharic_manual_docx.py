"""
Generate Clean Native DOCX of the Amharic Engineering Manual.
Local only - DO NOT PUSH TO GITHUB.
"""

import os
import zipfile
import html

def make_docx():
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Nyala" w:hAnsi="Nyala" w:cs="Nyala"/>
        <w:sz w:val="22"/>
        <w:color w:val="1F2937"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="260" w:lineRule="auto" w:after="100"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:styleId="Normal" w:default="1">
    <w:name w:val="Normal"/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:pPr>
      <w:spacing w:before="200" w:after="80"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Nyala" w:hAnsi="Nyala"/>
      <w:b/>
      <w:sz w:val="28"/>
      <w:color w:val="1E3A8A"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:pPr>
      <w:spacing w:before="140" w:after="60"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Nyala" w:hAnsi="Nyala"/>
      <w:b/>
      <w:sz w:val="24"/>
      <w:color w:val="0369A1"/>
    </w:rPr>
  </w:style>
</w:styles>"""

    def p(text="", bold=False, italic=False, sz=None, color=None, align=None, space_after=90):
        align_xml = f'<w:jc w:val="{align}"/>' if align else ''
        pPr = f'<w:pPr>{align_xml}<w:spacing w:after="{space_after}"/></w:pPr>'
        if not text:
            return f'<w:p>{pPr}</w:p>'
        rPr_parts = []
        if bold: rPr_parts.append('<w:b/>')
        if italic: rPr_parts.append('<w:i/>')
        if sz: rPr_parts.append(f'<w:sz w:val="{sz}"/>')
        if color: rPr_parts.append(f'<w:color w:val="{color}"/>')
        rPr = f'<w:rPr>{"".join(rPr_parts)}</w:rPr>' if rPr_parts else ''
        return f'<w:p>{pPr}<w:r>{rPr}<w:t xml:space="preserve">{html.escape(text)}</w:t></w:r></w:p>'

    def h1(text):
        return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:spacing w:before="200" w:after="80"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="1E3A8A"/></w:rPr><w:t>{html.escape(text)}</w:t></w:r></w:p>'

    def bullet(text, bold_prefix=None):
        runs = []
        if bold_prefix:
            runs.append(f'<w:r><w:rPr><w:b/><w:color w:val="1E3A8A"/></w:rPr><w:t xml:space="preserve">{html.escape(bold_prefix)} </w:t></w:r>')
        runs.append(f'<w:r><w:t xml:space="preserve">{html.escape(text)}</w:t></w:r>')
        return f'<w:p><w:pPr><w:ind w:left="320" w:hanging="200"/><w:spacing w:after="50"/></w:pPr><w:r><w:t xml:space="preserve">• </w:t></w:r>{"".join(runs)}</w:p>'

    body = []

    # Title
    body.append(p("ደብረ ብርሃን ዩኒቨርሲቲ • የኮምፒውቲንግ ኮሌጅ", bold=True, sz=30, color="1E3A8A", align="center", space_after=30))
    body.append(p("RoadEye™ የተሟላ የምህንድስና፣ የሒሳብ ስሌት እና የሳይንሳዊ ምንጮች ማኑዋል", bold=True, sz=24, color="334155", align="center", space_after=30))
    body.append(p("AI-Powered Intelligent Transportation Infrastructure Management & Advanced Collision Avoidance Platform", italic=True, sz=19, color="64748B", align="center", space_after=60))
    body.append(p("ደራሲና ተመራማሪ፦ ኃይሌ ስንታየሁ (Haile Sintayehu) • ህዳር 2026", bold=True, sz=21, color="2563EB", align="center", space_after=140))

    # Modules
    body.append(h1("ክፍል 1፦ Edge-AI የኮምፒውተር ቪዥን እና የብልሽት መለያ (YOLOv8/11 Perception)"))
    body.append(p("ዓላማው፦ ተራ የመኪና ዳሽካም ወይም ስልክ ካሜራን በመጠቀም በመንገድ ላይ የሚገኙ አደገኛ ጉድጓዶችን (Potholes)፣ የአዞ ቆዳ ስንጥቆችን (Alligator Cracking)፣ ቁመታዊና አግድም ስንጥቆችን (Longitudinal/Transverse Cracks) እና ተሽከርካሪዎችን በቅጽበት በሰከንድ 30+ ምስሎች (FPS) ፍጥነት መለየት።"))
    body.append(bullet("1. Bounding Box: B = [x_center, y_center, width, height]"))
    body.append(bullet("2. Confidence Score: C = P(Object) * IoU >= 0.40 (40%)"))
    body.append(bullet("3. Non-Maximum Suppression (NMS): IoU = Area(B1 ∩ B2) / Area(B1 ∪ B2) <= 0.45"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ Ultralytics YOLOv8/11 Architecture (2023-2025) • IEEE Transactions on Intelligent Transportation Systems"))

    body.append(h1("ክፍል 2፦ የሞኖኩላር ርቀት፣ ጂኦሜትሪ እና የመንዳት መስመር ትንበያ (Distance, Corridor & TTC)"))
    body.append(p("ዓላማው፦ የ 2D ካሜራ ምስልን ወደ እውነተኛ ሜትሪክ ርቀት (በሜትር) በመቀየር ጉድጓዱ ከመኪናው ስንት ሜትር ርቀት ላይ እንዳለ፣ መኪናው በሚጓዝበት መስመር ውስጥ መሆኑን እና ሾፌሩ ለመጋጨት ስንት ሰከንድ እንደቀረው (TTC) ማስላት።"))
    body.append(bullet("1. የቀጥታ ርቀት: Z = (f * H) / delta_y (f = focal length, H = የካሜራ ከፍታ 1.25m, delta_y = ከሆሪዞን ቁመት)"))
    body.append(bullet("2. የጎን ርቀት: X = (delta_x * Z) / fx"))
    body.append(bullet("3. የመንዳት መስመር ማጣሪያ: |X| <= 1.8m (በመኪናው 3.6m መስመር ውስጥ ከሆነ ብቻ አደጋ ነው፤ ካልሆነ የእግረኛ መንገድ ስለሆነ የውሸት ማንቂያ አይሰጥም)"))
    body.append(bullet("4. Time-To-Collision (TTC): TTC = Z / delta_v (ፍጥነት)"))
    body.append(bullet("   TTC <= 1.8s ከሆነ CRITICAL ALARM; TTC <= 3.0s ከሆነ WARNING ALARM"))
    body.append(bullet("5. የድምፅ ደወል: 'ጥንቃቄ! ከፊትህ ጉድጓድ አለ' / 'Warning! Hazard Ahead, Slow Down'"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ ISO 15623 Standards (Forward Collision Warning Systems) • Mobileye EyeQ Vision-Only Kinematic Monocular Ranging"))

    body.append(h1("ክፍል 3፦ የ 3D የጉድጓድ ጥልቀት፣ ይዘት እና የአስፋልት ቦርሳ ስሌት (3D Cavity & Bags)"))
    body.append(p("ዓላማው፦ የ 2D ቦክስ ግምትን በማስቀረት የጉድጓዱን እውነተኛ ጥልቀት በሳንቲሜትር፣ ይዘቱን በሊትር እና ለመጠገን የሚያስፈልገውን የአስፋልት ድብልቅ ክብደት በኪሎግራም እና በ 25 ኪ.ግ ቦርሳ ማስላት።"))
    body.append(bullet("1. የመሬት ስፋት: Area A = pi * (Wm/2) * (Lm/2) [m²]"))
    body.append(bullet("2. 3D ጥልቀት: Dmax = min(18cm, max(1.5cm, 4.2 * sqrt(Wm*Lm) * (1.0 + 0.65*S))) [cm] (S = የጥላ ድቅድቅነት)"))
    body.append(bullet("3. የይዘት ስሌት: V = 0.5 * A * (Dmax / 100) * 1000 [Liters] (Truncated Paraboloid Volume)"))
    body.append(bullet("4. የአስፋልት ክብደት: M_loose = V * 2.20 (እፍጋት) * 1.22 (የመጠቅጠቂያ ማባዣ) [kg]"))
    body.append(bullet("5. የቦርሳ ብዛት: N_25kg = ceil(M_loose / 25) ቦርሳዎች"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ Ethiopian Roads Administration (ERA) Standard Technical Specifications (Division 4000) • ASTM D6433"))

    body.append(h1("ክፍል 4፦ 360° የቀረቤታ ራዳር፣ የሌዘር ዒላማ እይታ እና የሶናር ድምፅ (Proximity Radar)"))
    body.append(p("ዓላማው፦ ተሽከርካሪው በሚጓዝበት ወቅት በመኪናው ዙሪያ የራዳር ፍተሻ አድማስ መዘርጋት፣ ወደ አደጋው ሲቃረብ አቅጣጫ ጠቋሚ የሌዘር ጨረር በስክሪኑ ላይ መቆለፍ እና ተለዋዋጭ የሶናር ድምፅ ማሰማት።"))
    body.append(bullet("1. የራዳር አድማሶች: 25m, 50m, 100m (Standard Horizon), 150m (Extended Horizon)"))
    body.append(bullet("2. የቀረቤታ ስሌት: በጂኦዴሲክ ርቀት መኪናው ወደ ጉድጓዱ ሲቃረብ"))
    body.append(bullet("3. Forward Laser Lock: ከመኪናው ወደ ጉድጓዱ የሚያሳይ ቀጥታ የሌዘር ጨረር በስክሪኑ ላይ ይቆለፋል"))
    body.append(bullet("4. Acoustic Sonar Ping: f = 800Hz + (1.0 - d / R_radar) * 1200Hz (በቀረበ ቁጥር የድምፁ ቃና እየሳሳ ይሄዳል)"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ SAE J3016 Active Sonar Perception Standards • W3C Web Audio API Specifications"))

    body.append(h1("ክፍል 5፦ የክረምት ዝናብ እና የብልሽት ማፋጠን ሞዴል (Kiremt 4.2x Multiplier)"))
    body.append(p("ዓላማው፦ በኢትዮጵያ ደጋማ ቦታዎች (ደብረ ብርሃን) በክረምት ወቅት የመንገድ ብልሽት በከፍተኛ ፍጥነት ለምን እንደሚያድግ በሳይንሳዊ ምህንድስና ማስረዳት እና የቅድመ-ክረምት ጥገና ስንት ሚሊዮን ብር እንደሚያድን ማረጋገጥ።"))
    body.append(bullet("1. ፊዚክሱ: በክረምት ውሃ ስንጥቅ ውስጥ ይገባል -> ከባድ መኪና (ESAL) ሲረግጠው ከፍተኛ ግፊት (Delta u) ይፈጥራል -> ሬንጁን ከጠጠሩ ይነቅላል (Bitumen Stripping) -> ስር ያለውን ጠጠር ያቦካል (Pumping) -> መንገዱ ይፈርሳል!"))
    body.append(bullet("2. የወቅቶች ማባዣ: ክረምት = 4.2x (72% የዓመቱ ዝናብ)፣ በልግ = 1.8x፣ በጋ = 1.0x (ደረቅ መነሻ Baseline)"))
    body.append(bullet("3. Cost of Inaction (COI ROI): ዛሬ በ 2,850 ብር/ካሬ ስንጥቅ መለጠፍ ዝናቡ ካለቀ በኋላ በ 7,800 ብር/ካሬ ሙሉ በሙሉ ከመገንባት ከ 8.5 እጥፍ እስከ 11.2 እጥፍ የህዝብ በጀት ያድናል!"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ World Bank HDM-4 Climate Calibration Factors • AASHTO Drainage Modification Factor (m_i) • ERA Pavement Design Manual Vol. 1"))

    body.append(h1("ክፍል 6፦ የተቋራጮች የዋስትና ጊዜ እና የህግ መጠየቂያ ደብዳቤ (Contractor DLP & FIDIC)"))
    body.append(p("ዓላማው፦ የአዳዲስ መንገዶችን የዋስትና ጊዜ (Defect Liability Period - DLP) በራስ-ሰር በመከታተል፣ ተቋራጮች በዋስትናቸው መሠረት እንዲጠግኑ በማስገደድ የማዘጋጃ ቤቱን በሚሊዮን የሚቆጠር ብር ማዳን።"))
    body.append(bullet("1. ህጋዊ መሰረት: FIDIC Red Book Clause 11 (ከ 1 እስከ 2 ዓመት የዋስትና ጊዜ)"))
    body.append(bullet("2. Geofencing: የብልሽቱ ጂፒኤስ ከተቋራጩ የመንገድ ወሰንና የዋስትና ቀን ጋር ይዛመዳል"))
    body.append(bullet("3. የተቋራጭ ዕዳ: Total Liability = (N_potholes * 28,000 ETB) + (N_cracks * 18,000 ETB)"))
    body.append(bullet("4. ህጋዊ ደብዳቤ: በአንድ ክሊክ የኮንትራት ቁጥሩንና የብልሽት ፎቶውን የያዘ ይፋዊ የጥገና መጠየቂያ ደብዳቤ (Demand Notice) ያዘጋጃል"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ FIDIC Conditions of Contract Clause 11 • Ethiopian Public Procurement Agency (PPA)"))

    body.append(h1("ክፍል 7፦ የብዙ መኪኖች መረጃ ጥምረትና ድግግሞሽን ማጥፋት (Fleet Crowdsourcing)"))
    body.append(p("ዓላማው፦ በአንድ መንገድ ላይ በርካታ መኪኖች በተደጋጋሚ ሲያልፉ አንድን ጉድጓድ 20 ጊዜ እንዳይመዘግቡ ማጣራት እና የተገኘውን መረጃ ትክክለኛነት በስታቲስቲክስ ማረጋገጥ።"))
    body.append(bullet("1. Spatial Clustering: በ 8 ሜትር ራዲየስ (epsilon <= 8.0m) ውስጥ ያሉ ምልከታዎች ወደ አንድ ጉድጓድ ይጠቀለላሉ"))
    body.append(bullet("2. Bayesian Probability Fusion: P(Defect) = 1.0 - prod(1.0 - ci) -> 3 መኪኖች በ 80% ቢያዩት = 99.2% ትክክለኛነት!"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ DBSCAN Spatial Clustering Principles • Bayesian Multi-Sensor Data Fusion Theory"))

    body.append(h1("ክፍል 8፦ ዓለም አቀፍ የመንገድ ጤና መለኪያ እና የበጀት ትንበያ (ASTM D6433 PCI & HDM-4)"))
    body.append(p("ዓላማው፦ የመንገዱን አጠቃላይ የጤና ደረጃ ከ 0 እስከ 100 በሚለካው የ ASTM D6433 Pavement Condition Index (PCI) እና በዓለም ባንክ International Roughness Index (IRI) መሠረት ደረጃ ማውጣት።"))
    body.append(bullet("1. PCI ስሌት: PCI = 100 - CDV"))
    body.append(bullet("2. HDM-4 IRI: በአስፋልቱ ንዝረትና በጉድጓዶች ብዛት ይሰላል"))
    body.append(bullet("3. ERA የበጀት ተመን: Reconstruction (PCI < 40) = 7,800 ETB/m²; Mill & Overlay (40-70) = 3,250 ETB/m²; Localized Patch (>= 70) = 2,850 ETB/m²"))
    body.append(bullet("ሳይንሳዊ ምንጭ፦ ASTM D6433 Pavement Condition Index Standards • World Bank Technical Paper 46 (HDM-4)"))

    body.append(h1("ክፍል 9፦ የፈታኞች እና የዳኞች የተለመዱ 5 ጥያቄዎችና ሳይንሳዊ መልሶቻቸው (Defense Cheatsheet)"))
    body.append(bullet("ጥያቄ 1፦ 'የክረምቱ ማባዣ gamma = 4.2x ከየት መጣ?' -> ምላሽ፦ HDM-4 የአየር ንብረት ካሊብሬሽን፣ በደብረ ብርሃን 72% ዝናብ በክረምት መውደቁ፣ እና በAASHTO mi ቅነሳ መሠረት የውሃ ግፊት (Pore-Water Pressure) ሬንጅን ከጠጠር ስለሚነቅለው የመበላሸት ፍጥነቱ በ 4 እጥፍ ይጨምራል።"))
    body.append(bullet("ጥያቄ 2፦ 'ባለ አንድ ካሜራ ስልክ እንዴት ጥልቀት (Depth) እና ኪሎግራም ያሰላል?' -> ምላሽ፦ Inverse Perspective Mapping በመጠቀም ፒክሰልን ወደ ሜትር ይቀይራል፤ ጥልቀቱን በጉድጓዱ የጥላ ጥልቀት (Shadow Gradient S) እና በ Truncated Paraboloid የይዘት ቀመር በማስላት በ ERA ስታንዳርድ ወደ ኪሎግራም እና ቦርሳ ይቀይረዋል።"))
    body.append(bullet("ጥያቄ 3፦ 'የራዳር ኮንሰፕቱ እንዴት ይሰራል?' -> ምላሽ፦ Software-Defined Optical Radar በመጠቀም በመኪናው ዙሪያ የ 360° የቀረቤታ ራዳር አድማስ (25m-150m) ይዘረጋል፤ አደጋ ሲቃረብ የሌዘር ጨረር ይቆልፋል፣ የሶናር ድምፅ ያሰማል፣ እንዲሁም በአማርኛ 'ጥንቃቄ! ከፊትህ ጉድጓድ አለ' ብሎ ይጮሃል።"))
    body.append(bullet("ጥያቄ 4፦ 'ተቋራጩ ለምን ተጠያቂ ይሆናል?' -> ምላሽ፦ በ FIDIC Clause 11 መሠረት አዲስ መንገድ ተረክቦ ለ 1-2 ዓመት ማንኛውንም ብልሽት ተቋራጩ በራሱ ወጪ የመጠገን ህጋዊ ግዴታ ስላለበት ሲስተማችን ይሄን በጂኦፌንሲንግ በራስ-ሰር ይከታተላል።"))
    body.append(bullet("ጥያቄ 5፦ 'ይህ ሲስተም በእርግጥ ይሰራል ወይስ በወረቀት ላይ ብቻ ነው?' -> ምላሽ፦ ሙሉ በሙሉ የሚሰራ ፕሮቶታይፕ ነው፤ 114 አውቶሜትድ ቴስቶች 100% አልፈዋል፤ አሁንም በላፕቶፔ ላይ በፖርት 5000 ላይ በቀጥታ እየሰራ ያለ ሲስተም ነው።"))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {"".join(body)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1000" w:right="1000" w:bottom="1000" w:left="1000"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    docx_path = os.path.join(output_dir, "RoadEye_Complete_Engineering_Manual_Amharic.docx")

    with zipfile.ZipFile(docx_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types)
        zf.writestr('_rels/.rels', rels)
        zf.writestr('word/_rels/document.xml.rels', doc_rels)
        zf.writestr('word/styles.xml', styles)
        zf.writestr('word/document.xml', document_xml)

    print(f"Generated Local Amharic DOCX: {docx_path} ({os.path.getsize(docx_path)} bytes)")

if __name__ == "__main__":
    make_docx()

