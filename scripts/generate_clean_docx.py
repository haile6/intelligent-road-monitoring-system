"""
Generate Clean, Concise, Native Word (.docx) Business Proposal
Focused on In-Cabin Driver Voice Warnings & Municipal Infrastructure Intelligence
For Debre Berhan University & EDI Business Idea Competition 2026.
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
        <w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI" w:cs="Segoe UI"/>
        <w:sz w:val="21"/>
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
      <w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/>
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
      <w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/>
      <w:b/>
      <w:sz w:val="24"/>
      <w:color w:val="0369A1"/>
    </w:rPr>
  </w:style>
</w:styles>"""

    def p(text="", bold=False, italic=False, sz=None, color=None, align=None, space_after=100):
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

    def h2(text):
        return f'<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:spacing w:before="140" w:after="60"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="23"/><w:color w:val="0284C7"/></w:rPr><w:t>{html.escape(text)}</w:t></w:r></w:p>'

    def bullet(text, bold_prefix=None):
        runs = []
        if bold_prefix:
            runs.append(f'<w:r><w:rPr><w:b/><w:color w:val="1E3A8A"/></w:rPr><w:t xml:space="preserve">{html.escape(bold_prefix)} </w:t></w:r>')
        runs.append(f'<w:r><w:t xml:space="preserve">{html.escape(text)}</w:t></w:r>')
        return f'<w:p><w:pPr><w:ind w:left="320" w:hanging="200"/><w:spacing w:after="50"/></w:pPr><w:r><w:t xml:space="preserve">• </w:t></w:r>{"".join(runs)}</w:p>'

    def table(headers, rows):
        tbl_parts = ['<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/></w:tblBorders></w:tblPr>']
        tbl_parts.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>')
        for h in headers:
            tbl_parts.append(f'<w:tc><w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="1E3A8A"/><w:tcMar><w:top w:w="100" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:left w:w="120" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tcMar></w:tcPr><w:p><w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/><w:sz w:val="19"/></w:rPr><w:t>{html.escape(h)}</w:t></w:r></w:p></w:tc>')
        tbl_parts.append('</w:tr>')
        for r_idx, r in enumerate(rows):
            bg = 'F8FAFC' if r_idx % 2 == 1 else 'FFFFFF'
            tbl_parts.append('<w:tr>')
            for cell in r:
                tbl_parts.append(f'<w:tc><w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="{bg}"/><w:tcMar><w:top w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:left w:w="120" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tcMar></w:tcPr><w:p><w:r><w:rPr><w:sz w:val="19"/><w:color w:val="334155"/></w:rPr><w:t>{html.escape(str(cell))}</w:t></w:r></w:p></w:tc>')
            tbl_parts.append('</w:tr>')
        tbl_parts.append('</w:tbl>')
        return "".join(tbl_parts)

    body = []

    # Header Box
    body.append(p("DEBRE BERHAN UNIVERSITY", bold=True, sz=30, color="1E3A8A", align="center", space_after=30))
    body.append(p("Center for Employability and Entrepreneurship Development", bold=True, sz=22, color="334155", align="center", space_after=30))
    body.append(p("In Collaboration with Entrepreneurship Development Institute (EDI)", italic=True, sz=19, color="64748B", align="center", space_after=80))
    body.append(p("BUSINESS IDEA COMPETITION PROPOSAL", bold=True, sz=26, color="2563EB", align="center", space_after=30))
    body.append(p("Global Entrepreneurship Week 2026", sz=19, color="475569", align="center", space_after=140))

    # Meta Table
    body.append(table(
        ["Field", "Information", "Field", "Details"],
        [
            ["Submission Deadline", "November 16, 2026 – 5:00 PM", "Host Office", "Center for Employability / SE-02, F2"],
            ["Submission Email", "deredbu2003@gmail.com", "Submission Format", "Word / PDF Document"],
            ["Lead Innovator", "Haile Sintayehu", "College / Dept", "College of Computing"],
            ["Applicant Email", "hailesintayehu113@gmail.com", "Phone Contact", "0921746222 / 0912485346"],
            ["Target Focus Areas", "ICT • Infrastructure Materials • Engineering", "Prototype Status", "114 Automated Unit Tests Passing (100%)"]
        ]
    ))
    body.append(p("", space_after=100))

    # Title Banner
    body.append(p("RoadEye™: In-Cabin Driver Voice Warning & Municipal Asset Management Platform", bold=True, sz=24, color="1E3A8A", align="center", space_after=30))
    body.append(p("Dual-Action: Real-Time Collision Avoidance (TTC) & Autonomous Infrastructure Maintenance", italic=True, sz=19, color="0284C7", align="center", space_after=140))

    # 1. Title and Description
    body.append(h1("1. Business Idea Title and Description"))
    body.append(p("Title: RoadEye™: Edge-AI In-Cabin Driver Safety Voice Alarm & Municipal Asset Management Platform", bold=True, color="1E3A8A"))
    body.append(p("RoadEye™ is an innovative GovTech and Mobility AI platform engineered at Debre Berhan University. It converts affordable dashcams and smartphones into a dual-action system:"))
    body.append(bullet("Scans the forward driving corridor, calculates closing Time-To-Collision (TTC <= 1.8s), and triggers immediate voice alarms in both Amharic and English ('ጥንቃቄ! ከፊትህ ጉድጓድ አለ' / 'Warning! Pothole Ahead, Slow Down') to prevent rollover crashes, tire blowouts, and sudden swerving.", "1. For Drivers (In-Cabin ADAS Safety):"))
    body.append(bullet("Automatically maps pavement distresses, computes 3D cavity depth and required asphalt repair mass (kg), projects defects onto a GIS heatmap, and enforces contractor road warranties (FIDIC Clause 11 DLP).", "2. For Municipalities (GovTech Asset Management):"))
    body.append(p("By identifying defects before the Ethiopian monsoon (Kiremt, where road damage accelerates by 4.2x), RoadEye™ cuts municipal road repair costs by up to 75% while saving driver lives daily."))

    # 2. Problem Statement
    body.append(h1("2. Problem Statement and Market Need"))
    body.append(bullet("High-speed drivers suddenly encounter deep potholes, causing violent swerving into oncoming traffic, vehicle rollovers, and fatal collisions. Without automated in-cabin voice warnings, drivers cannot react safely at night or in rain.", "1. Lethal Driver Collisions from Sudden Potholes:"))
    body.append(bullet("Technicians walk along high-speed corridors with tape measures and clipboards. It is lethally dangerous, covers under 5% of the network annually, and takes months to compile.", "2. Inefficient & Dangerous Manual Inspections:"))
    body.append(bullet("Water infiltration combined with heavy truck freight on Route A2 accelerates pavement deterioration by 4.2x during Kiremt, turning hairline cracks into severe potholes in days.", "3. Severe Monsoon (Kiremt) Pavement Destruction:"))
    body.append(bullet("When newly built roads fail during the 1-2 year warranty period (FIDIC Clause 11 DLP), municipalities lack digital evidence, forcing taxpayers to pay tens of millions of ETB that contractors should fix for free.", "4. Millions Lost from Unenforced Contractor Warranties:"))
    body.append(bullet("Foreign laser inspection vans (ROMDAS, Dynatest) cost $250,000-$500,000 USD, creating an impossible foreign exchange barrier for Ethiopian cities.", "5. Foreign Currency (Forex) Crisis:"))

    # 3. Proposed Solution
    body.append(h1("3. Proposed Product and Technical Solution"))
    body.append(bullet("Monitors the forward travel lane at 30+ FPS and issues dual-language auditory voice warnings ('ጥንቃቄ! ከፊትህ ጉድጓድ አለ') and flashing visual HUD alarms when TTC <= 1.8 seconds, giving drivers vital reaction time.", "A. Active Driver Voice Announcement & ADAS Alarm:"))
    body.append(bullet("Calculates maximum depth (cm), cavity volume (Liters), and exact commercial asphalt mix mass (in kg and 25kg bags) compliant with ERA specifications.", "B. Monocular 3D Cavity & Asphalt Quantification:"))
    body.append(bullet("Clusters observations from multiple vehicles within 8 meters, eliminating duplicate reports and achieving 99.9% statistical certainty.", "C. Multi-Vehicle Fleet Crowdsourcing:"))
    body.append(bullet("Cross-references defect coordinates with contractor warranty zones and unit rates (28,000 ETB/pothole, 18,000 ETB/crack) to produce formal legal rectification demand notices.", "D. Contractor Warranty Engine (FIDIC Clause 11):"))
    body.append(bullet("Employs a 5-state Markov chain model to prove that spending 2,850 ETB/m² on proactive crack sealing today saves 7,800 ETB/m² in emergency reconstruction after Kiremt (8.5x-11.2x ROI).", "E. Rainy Season Cost of Inaction (COI) ROI Engine:"))

    # 4. Target Customers & Market Drivers
    body.append(h1("4. Target Customers, Market Potential & Key Drivers"))
    body.append(bullet("Debre Berhan City Administration, Zonal & Regional Urban Infrastructure Bureaus.", "1. Municipalities (GovTech B2G):"))
    body.append(bullet("Ethiopian Roads Administration (ERA) and Regional Road Bureaus.", "2. National Highway Authorities:"))
    body.append(bullet("Defence Construction, Sur Construction, Midroc, Yencomad (for quality audits & warranty defense).", "3. Road Construction Contractors (B2B):"))
    body.append(bullet("Freight trucks on Route A2, bus fleets, and logistics companies requiring in-cabin driver alerts.", "4. Commercial Transport Fleets:"))
    body.append(h2("Market Size (TAM / SAM / SOM)"))
    body.append(bullet("TAM: 160,000+ km classified Ethiopian road network; annual national maintenance budget > 25 Billion ETB."))
    body.append(bullet("SAM: Amhara Region & Central Highway Corridors (18,500+ km) valued at 180 Million ETB/year."))
    body.append(bullet("SOM: Debre Berhan road network (180+ km) and adjacent A2 corridor, valued at 12-18 Million ETB/year."))
    body.append(h2("Key Market Drivers (Why Customers Buy Now)"))
    body.append(bullet("MUI and ERA now require objective digital Pavement Condition Index (PCI) logs before approving annual municipal road maintenance budgets.", "1. Federal Road Fund Digital Mandate:"))
    body.append(bullet("The Federal Auditor General (OFAG) holds municipal officials personally liable if they pay for repairs that contractors should fix for free under warranty.", "2. Strict Anti-Corruption & Public Procurement Audits:"))
    body.append(bullet("Rapid asphalt expansion in Debre Berhan and regional cities creates billions of ETB in new road assets entering their critical 1-2 year warranty periods.", "3. Urban Corridor Development Boom:"))
    body.append(bullet("Foreign exchange shortages make imported laser survey vans ($500k USD) unaffordable, driving urgent demand for 100% locally built GovTech solutions.", "4. Forex Crunch & Import Substitution Policy:"))
    body.append(bullet("Asphalt bitumen and fuel inflation (>250%) makes full reconstruction economically unsustainable, forcing municipalities to invest in proactive crack sealing.", "5. Construction Material Inflation:"))

    # 5. Competitive Advantage
    body.append(h1("5. Innovation and Competitive Advantage"))
    body.append(table(
        ["Feature", "Manual Survey", "Imported Laser Vans", "RoadEye™ (Our Innovation)"],
        [
            ["Hardware Cost", "Low", "$250,000 – $500,000 USD", "< $350 USD (Standard dashcam / phone)"],
            ["Forex Dependency", "None", "100% Forex Dependent", "0% Forex (100% locally built & hosted)"],
            ["Driver Voice Alarm", "None", "None", "Real-Time Dual-Language (አማርኛ / English)"],
            ["Survey Speed", "3 – 5 km/day", "40 – 60 km/day", "100 – 300+ km/day"],
            ["Worker Safety", "Extremely High Risk", "Low Risk", "Zero Risk (Surveyed from vehicle cabin)"],
            ["3D Cavity Material", "Guesswork", "Requires LiDAR", "Automated Monocular 3D (Bags & kg)"],
            ["Contractor DLP", "None", "Manual / None", "Automated FIDIC Clause 11 Geofencing"],
            ["Current Status", "Legacy", "Commercial", "Working Prototype (114 Passing Tests)"]
        ]
    ))
    body.append(p("", space_after=80))

    # 6. Business Model
    body.append(h1("6. Business Model and Commercialization Strategy"))
    body.append(bullet("Tiered annual licensing: Basic (250,000 ETB), Pro / Debre Berhan (650,000 ETB), Enterprise Metro (1,800,000 ETB).", "1. Municipal SaaS Subscriptions:"))
    body.append(bullet("Turnkey 3D surveying at 1,200–1,800 ETB per kilometer.", "2. Pavement Inspection-as-a-Service (PIaaS):"))
    body.append(bullet("Pre-handover audit reports at 45,000–120,000 ETB per project.", "3. Contractor DLP Compliance Audits:"))
    body.append(bullet("Monthly subscription for in-cabin driver alerts and route hazard telemetry.", "4. Fleet Safety & Telemetry Feeds:"))
    body.append(p("Three-Year Financial Forecast:", bold=True))
    body.append(table(
        ["Year", "Strategic Milestone", "Clients / Projects", "Projected Revenue", "Net Margin"],
        [
            ["2027", "Debre Berhan Pilot & Fleet Launch", "1 Municipality + 4 Contractors", "2,450,000 ETB", "42%"],
            ["2028", "Expansion to 5 Amhara Cities + ERA", "6 Municipalities + 12 Contractors", "8,900,000 ETB", "56%"],
            ["2029", "National Corridor Expansion", "18 Municipalities + 35 Contractors", "26,500,000 ETB", "68%"]
        ]
    ))

    # 7. Impact
    body.append(h1("7. Expected Economic, Social, and Safety Impact"))
    body.append(bullet("Real-time voice announcements ('ጥንቃቄ! ከፊትህ ጉድጓድ አለ') prevent violent swerving, rollovers, and multi-vehicle highway collisions.", "Saving Driver Lives & Preventing Crashes:"))
    body.append(bullet("Proactive crack sealing and contractor warranty enforcement save 15-30 Million ETB annually for Debre Berhan City Administration.", "Massive Municipal Budget Savings:"))
    body.append(bullet("Replaces scarce foreign currency imports with locally engineered software and standard commercial hardware.", "Preserving Foreign Currency:"))
    body.append(bullet("Creates high-skill employment in AI engineering, computer vision, GIS analysis, and municipal infrastructure management in Debre Berhan.", "Youth Employment & Tech Innovation:"))

    # 8. Team
    body.append(h1("8. Team Information and Contact Details"))
    body.append(table(
        ["Parameter", "Details"],
        [
            ["Lead Innovator", "Haile Sintayehu"],
            ["Position", "Lead AI & Computer Vision Developer / System Architect"],
            ["Institution / College", "Debre Berhan University – College of Computing"],
            ["Email Address", "hailesintayehu113@gmail.com"],
            ["Contact Numbers", "0921746222 / 0912485346"],
            ["GitHub Portfolio", "github.com/haile6"],
            ["Project Repository", "haile6/intelligent-road-monitoring-system"]
        ]
    ))
    body.append(p("", space_after=100))
    body.append(p("Declaration: I submit this business proposal to the Debre Berhan University Center for Employability and Entrepreneurship Development and EDI for the Global Entrepreneurship Week 2026 Business Idea Competition. RoadEye™ is an original innovation developed at Debre Berhan University, validated by 114 passing unit tests, ready for pilot deployment and commercial incubation.", italic=True))
    body.append(p("", space_after=60))
    body.append(p("Applicant Signature: Haile Sintayehu                         Date: November 16, 2026", bold=True))

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
    docx_path = os.path.join(output_dir, "DBU_EDI_Business_Idea_Proposal_RoadEye.docx")

    with zipfile.ZipFile(docx_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types)
        zf.writestr('_rels/.rels', rels)
        zf.writestr('word/_rels/document.xml.rels', doc_rels)
        zf.writestr('word/styles.xml', styles)
        zf.writestr('word/document.xml', document_xml)

    print(f"Generated Clean DOCX: {docx_path} ({os.path.getsize(docx_path)} bytes)")

if __name__ == "__main__":
    make_docx()
