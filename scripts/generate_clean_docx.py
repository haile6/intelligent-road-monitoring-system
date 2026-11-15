import os
import zipfile
import html

def make_docx():
    # Define XML components
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
        <w:sz w:val="22"/>
        <w:color w:val="1F2937"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="276" w:lineRule="auto" w:after="120"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:styleId="Normal" w:default="1">
    <w:name w:val="Normal"/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/>
      <w:b/>
      <w:sz w:val="32"/>
      <w:color w:val="1E3A8A"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:pPr>
      <w:spacing w:before="180" w:after="80"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/>
      <w:b/>
      <w:sz w:val="26"/>
      <w:color w:val="0369A1"/>
    </w:rPr>
  </w:style>
</w:styles>"""

    # Helper functions for generating WordprocessingML
    def p(text="", bold=False, italic=False, sz=None, color=None, align=None, space_after=120):
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
        escaped_text = html.escape(text)
        return f'<w:p>{pPr}<w:r>{rPr}<w:t xml:space="preserve">{escaped_text}</w:t></w:r></w:p>'

    def h1(text):
        escaped = html.escape(text)
        return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:spacing w:before="240" w:after="100"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="30"/><w:color w:val="1E3A8A"/></w:rPr><w:t>{escaped}</w:t></w:r></w:p>'

    def h2(text):
        escaped = html.escape(text)
        return f'<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:spacing w:before="160" w:after="60"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="0284C7"/></w:rPr><w:t>{escaped}</w:t></w:r></w:p>'

    def bullet(text, bold_prefix=None):
        runs = []
        if bold_prefix:
            runs.append(f'<w:r><w:rPr><w:b/><w:color w:val="1E3A8A"/></w:rPr><w:t xml:space="preserve">{html.escape(bold_prefix)} </w:t></w:r>')
        runs.append(f'<w:r><w:t xml:space="preserve">{html.escape(text)}</w:t></w:r>')
        return f'<w:p><w:pPr><w:ind w:left="360" w:hanging="240"/><w:spacing w:after="60"/></w:pPr><w:r><w:t xml:space="preserve">• </w:t></w:r>{"".join(runs)}</w:p>'

    def table(headers, rows):
        tbl_parts = ['<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/></w:tblBorders></w:tblPr>']
        
        # Header Row
        tbl_parts.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>')
        for h in headers:
            tbl_parts.append(f'<w:tc><w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="1E3A8A"/><w:tcMar><w:top w:w="120" w:type="dxa"/><w:bottom w:w="120" w:type="dxa"/><w:left w:w="140" w:type="dxa"/><w:right w:w="140" w:type="dxa"/></w:tcMar></w:tcPr><w:p><w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/><w:sz w:val="19"/></w:rPr><w:t>{html.escape(h)}</w:t></w:r></w:p></w:tc>')
        tbl_parts.append('</w:tr>')

        # Data Rows
        for r_idx, r in enumerate(rows):
            bg = 'F8FAFC' if r_idx % 2 == 1 else 'FFFFFF'
            tbl_parts.append('<w:tr>')
            for cell in r:
                tbl_parts.append(f'<w:tc><w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="{bg}"/><w:tcMar><w:top w:w="100" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:left w:w="140" w:type="dxa"/><w:right w:w="140" w:type="dxa"/></w:tcMar></w:tcPr><w:p><w:r><w:rPr><w:sz w:val="19"/><w:color w:val="334155"/></w:rPr><w:t>{html.escape(str(cell))}</w:t></w:r></w:p></w:tc>')
            tbl_parts.append('</w:tr>')
            
        tbl_parts.append('</w:tbl>')
        return "".join(tbl_parts)

    body = []

    # Title & Header Block
    body.append(p("DEBRE BERHAN UNIVERSITY", bold=True, sz=32, color="1E3A8A", align="center", space_after=40))
    body.append(p("Center for Employability and Entrepreneurship Development", bold=True, sz=24, color="334155", align="center", space_after=40))
    body.append(p("In collaboration with Entrepreneurship Development Institute (EDI)", italic=True, sz=20, color="64748B", align="center", space_after=120))
    body.append(p("BUSINESS IDEA COMPETITION PROPOSAL", bold=True, sz=28, color="2563EB", align="center", space_after=40))
    body.append(p("Organized in celebration of Global Entrepreneurship Week 2026", sz=20, color="475569", align="center", space_after=200))

    # Meta Table
    body.append(table(
        ["Field", "Submission Information", "Category", "Details"],
        [
            ["Submission Deadline", "November 16, 2026 – 5:00 PM", "Host Office", "Center for Employability / SE-02, F2"],
            ["Submission Email", "deredbu2003@gmail.com", "Format", "Word / PDF Format"],
            ["Lead Innovator", "Haile Sintayehu", "College/Dept", "College of Computing / Engineering"],
            ["Applicant Email", "hailesintayehu113@gmail.com", "Phone Contact", "0921746222 / 0912485346"],
            ["Target Focus Areas", "ICT • Construction & Infrastructure Materials • Engineering Technologies", "Working Demo", "114 Automated Unit Tests Passing (100%)"]
        ]
    ))
    body.append(p("", space_after=150))

    # Project Banner
    body.append(p("RoadEye™: AI-Powered Road Infrastructure Intelligence & Municipal Asset Management Platform", bold=True, sz=26, color="1E3A8A", align="center", space_after=40))
    body.append(p("Autonomous Pavement Defect Perception, 3D Cavity Quantification, Predictive Rainy Season Forecasting, and Contractor Warranty Enforcement", italic=True, sz=20, color="0284C7", align="center", space_after=180))

    # Section 1
    body.append(h1("1. Business Idea Title and Description"))
    body.append(p("Business Idea Title: RoadEye™: Edge-AI Road Infrastructure Intelligence & Municipal Asset Management SaaS", bold=True, color="1E3A8A"))
    body.append(p("RoadEye™ is an innovative enterprise GovTech and mobility venture founded at Debre Berhan University. We transform ordinary, affordable monocular cameras (dashcams, smartphones, and municipal patrol vehicle cameras) into intelligent, autonomous road survey systems."))
    body.append(p("Powered by customized edge computer vision (YOLOv8/11 deep learning), photogrammetric 3D depth geometry, spatial GIS kinematics, and physics-informed deterioration forecasting, RoadEye™ completely eliminates manual, slow, and subjective road inspections. In real-time, the platform detects pavement distresses (potholes, alligator cracking, rutting), estimates 3D cavity depth and required asphalt volume in kilograms, projects hazards on an interactive GIS municipal map, and enforces contractor road warranties (under FIDIC Clause 11 Defect Liability Period)."))
    body.append(p("By identifying road failures before the heavy Ethiopian monsoon (Kiremt) season accelerates their degradation by 4.2×, RoadEye™ enables Debre Berhan City Administration and road authorities to transition from expensive post-collapse emergency reconstruction to proactive, cost-saving preventive maintenance—saving municipalities up to 75% in annual road maintenance expenditures."))

    # Section 2
    body.append(h1("2. Problem or Market Need Addressed"))
    body.append(p("Ethiopian municipalities, the Ethiopian Roads Administration (ERA), and regional transport corridors (particularly the high-volume Route A2 Addis Ababa–Debre Berhan–Dessie corridor) suffer from a severe, costly infrastructure crisis:"))
    body.append(bullet("Technicians walk along high-speed corridors with measuring wheels and clipboards. This is subjective, lethally dangerous, covers under 5% of the network annually, and takes months to report.", "1. Dangerous & Manual Pavement Inspections:"))
    body.append(bullet("In the Ethiopian central highlands, continuous rain combined with heavy freight axle loads creates hydrodynamic pore-water pumping that strips bitumen from aggregates. Road deterioration accelerates 4.2× faster in Kiremt than in Bega, turning hairline cracks into severe potholes within 2 weeks.", "2. Catastrophic Monsoon (Kiremt) Acceleration:"))
    body.append(bullet("When new roads are constructed, contractors are legally bound under FIDIC Clause 11 (Defect Liability Period - DLP) to repair defects at their own expense for 1 to 2 years. Due to the lack of automated tracking, municipal offices fail to document defects before warranty expiration. As a result, municipal taxpayers end up paying tens of millions of ETB for repairs contractors were legally obligated to fix.", "3. Public Fund Losses from Unenforced Contractor Warranties:"))
    body.append(bullet("Substandard road surfaces and sudden, deep potholes are major contributing factors to rollover accidents, tire blowouts, and vehicle suspension damage in Debre Berhan, leading to tragic loss of lives and freight delays.", "4. Devastating Traffic Accidents & Freight Losses:"))
    body.append(bullet("Traditional foreign laser survey vans (ROMDAS, Dynatest) cost between $250,000 and $500,000 USD to purchase and maintain—completely unaffordable for Ethiopian municipal governments facing severe foreign currency shortages.", "5. Prohibitive Cost of Imported Inspection Vehicles:"))

    # Section 3
    body.append(h1("3. Proposed Product, Service, or Solution"))
    body.append(p("RoadEye™ delivers an affordable, end-to-end, locally engineered digital ecosystem consisting of three integrated layers:"))
    body.append(bullet("Low-cost dashcam or smartphone running real-time 30+ FPS YOLOv8 computer vision perception under diverse lighting and weather conditions.", "Layer 1: Edge Mobile Sensor Layer —"))
    body.append(bullet("Monocular 3D depth estimator (cm), truncated paraboloid cavity volume (Liters), asphalt patch mix mass (kg & 25kg bags) compliant with ERA specifications, and 5-state Markov deterioration forecasting.", "Layer 2: Physics & 3D Geometry Engine —"))
    body.append(bullet("Real-time GIS municipal dashboard, ASTM D6433 Pavement Condition Index (PCI), contractor warranty geofencing (FIDIC Clause 11), and automated legal rectification demand notices.", "Layer 3: Enterprise Web GIS Municipal Platform —"))
    body.append(p("Key Capabilities Include:"))
    body.append(bullet("Fleet crowdsourcing clusters observations within 8 meters, eliminating duplicate reports and boosting detection certainty to 99.9%.", "Multi-Vehicle Bayesian Fusion:"))
    body.append(bullet("Cross-references detected defects with contractor warranty dates and computes total contractor financial liabilities (e.g. 28,000 ETB/pothole, 18,000 ETB/crack) to produce formal rectification demand notices.", "Contractor Warranty Liability Engine:"))
    body.append(bullet("Proves that spending 2,850 ETB/m² on proactive sealing today prevents 7,800 ETB/m² emergency reconstruction in 90 days (delivering an 8.5× to 11.2× return on investment).", "Rainy Season Cost of Inaction (COI) ROI Engine:"))

    # Section 4
    body.append(h1("4. Target Customers and Market Potential"))
    body.append(h2("4.1 Target Customer Segments"))
    body.append(bullet("Debre Berhan City Administration (Infrastructure and Construction Department), Zonal and Regional Urban Planning Bureaus (Amhara Region, Oromia, Addis Ababa).", "1. Municipalities & City Administrations (GovTech B2G):"))
    body.append(bullet("Ethiopian Roads Administration (ERA) – Directorate of Road Asset Management and Maintenance.", "2. National Highway Authorities:"))
    body.append(bullet("Major civil contractors (Defence Construction Enterprise, Sur Construction, Midroc, Yencomad, regional builders) needing automated pre-handover and warranty inspection tools.", "3. General Road Construction Contractors (B2B):"))
    body.append(bullet("Bus fleets, freight operators on Route A2, and logistics companies requiring active in-cabin collision alerts and optimal road-quality routing.", "4. Public Transport & Logistics Fleets:"))
    body.append(bullet("Risk assessment data to settle vehicular suspension/tire damage claims and verify accident causation.", "5. Automotive Insurance Underwriters:"))
    body.append(h2("4.2 Market Size & Commercial Potential"))
    body.append(bullet("Over 160,000 km of classified federal and municipal road networks across Ethiopia, with an annual national road maintenance budget exceeding 25 Billion ETB.", "Total Addressable Market (TAM):"))
    body.append(bullet("Amhara Region & Central Highway Corridors covering 18,500+ km, representing an estimated annual inspection and condition monitoring market of 180 Million ETB.", "Serviceable Available Market (SAM):"))
    body.append(bullet("Debre Berhan City road network (180+ km paved roads) and the adjacent 130 km Debre Berhan–Addis Ababa freight corridor, with an immediate addressable annual value of 12 to 18 Million ETB.", "Immediate Serviceable Obtainable Market (SOM):"))
    body.append(h2("4.3 Key Market Potential Drivers (Why Customers Buy Now)"))
    body.append(bullet("MUI and ERA now require objective digital Pavement Condition Index (PCI) logs before approving annual municipal road maintenance budgets.", "1. Regulatory Mandates & Road Fund Requirements:"))
    body.append(bullet("The Federal Auditor General (OFAG) now penalizes municipal heads who fail to enforce contractor Defect Liability Period (DLP) obligations under FIDIC Clause 11, creating urgent demand for our automated warranty audit records.", "2. Strict Anti-Corruption & Public Procurement Audits:"))
    body.append(bullet("Debre Berhan, Adama, Hawassa, and regional capitals are rapidly expanding asphalt corridors, creating billions of ETB in new road assets entering critical 1-2 year warranty periods.", "3. Nationwide Urban Corridor Development Boom:"))
    body.append(bullet("Severe foreign exchange shortages make imported laser survey vans ($250k–$500k USD) unaffordable, driving urgent demand for 100% locally built GovTech solutions.", "4. Forex Crunch & Import Substitution Policy:"))
    body.append(bullet("Asphalt bitumen and fuel inflation (>250%) makes full reconstruction economically unsustainable, forcing municipalities to invest in proactive crack sealing prior to monsoon rains.", "5. Construction Material Inflation & Budget Crisis:"))

    # Section 5
    body.append(h1("5. Innovation and Competitive Advantage"))
    body.append(table(
        ["Feature / Metric", "Traditional Manual Survey", "Imported Laser Vans (ROMDAS)", "RoadEye™ (Our Innovation)"],
        [
            ["Hardware Cost", "Low (Tape & chalk)", "$250,000 – $500,000 USD", "< $350 USD (Standard dashcam / phone)"],
            ["Foreign Exchange", "None", "100% Forex dependent", "0% Forex (100% locally built & hosted)"],
            ["Survey Speed", "3 – 5 km / day (Walking)", "40 – 60 km / day", "100 – 300+ km / day (Vehicle speed)"],
            ["Worker Safety Risk", "Extremely High (Hit-and-run)", "Low", "Zero Risk (Surveyed from cabin)"],
            ["3D Cavity Material", "Manual guesswork", "Requires expensive LiDAR", "Automated Monocular 3D (Bags & kg)"],
            ["Contractor Geofencing", "Non-existent", "Manual export / None", "Native FIDIC Clause 11 Automation"],
            ["Weather Adaptation", "None", "Generic Western models", "Ethiopian Seasons (Kiremt/Belg/Bega)"],
            ["Current Status", "Legacy", "Commercial", "Fully Functional Prototype (114 Tests)"]
        ]
    ))
    body.append(p("", space_after=100))
    body.append(bullet("We have a fully functional web platform running live on http://127.0.0.1:5000 with 114 automated tests passing 100%.", "1. Working, Battle-Tested Prototype:"))
    body.append(bullet("Eliminates foreign exchange constraints for Ethiopian municipal administrations.", "2. Zero Foreign Currency Barrier:"))
    body.append(bullet("Direct integration with Ethiopian Roads Administration Schedule of Rates and FIDIC procurement conditions.", "3. Native Alignment with ERA Standards:"))

    # Section 6
    body.append(h1("6. Business Model and Commercialization Potential"))
    body.append(p("RoadEye™ deploys a high-margin, scalable hybrid SaaS (Software-as-a-Service) and DaaS (Data-as-a-Service) business model:"))
    body.append(table(
        ["Revenue Stream", "Target Customer", "Pricing Structure"],
        [
            ["1. Municipal SaaS Subscription", "City Administrations & Municipalities", "Annual: Basic (250,000 ETB), Pro / Debre Berhan (650,000 ETB), Enterprise Metro (1,800,000 ETB)"],
            ["2. Pavement Inspection-as-a-Service (PIaaS)", "Regional Road Bureaus & ERA", "1,200 to 1,800 ETB per kilometer scanned (turnkey 3D CAD/GIS report)"],
            ["3. Contractor DLP Compliance Audit Reports", "Road Construction Contractors", "45,000 – 120,000 ETB per project audit prior to final retention release"],
            ["4. Fleet & Telemetry API Feeds", "Transport Logistics & Insurance", "Monthly API access fee for live hazard coordinates & route quality"]
        ]
    ))
    body.append(p("", space_after=100))
    body.append(p("3-Year Financial Forecast:", bold=True))
    body.append(table(
        ["Year", "Milestone", "Clients / Projects", "Projected Revenue (ETB)", "Net Profit Margin"],
        [
            ["Year 1 (2027)", "Debre Berhan Pilot + 3 Local Contractors", "1 Municipality + 4 Contractors", "2,450,000 ETB", "42%"],
            ["Year 2 (2028)", "Expansion to 5 Amhara Region Cities + ERA", "6 Municipalities + 12 Contractors", "8,900,000 ETB", "56%"],
            ["Year 3 (2029)", "National Expansion (Addis, Hawassa, Adama)", "18 Municipalities + 35 Contractors", "26,500,000 ETB", "68%"]
        ]
    ))

    # Section 7
    body.append(h1("7. Expected Economic and Social Impact"))
    body.append(bullet("By enforcing contractor warranty liabilities (DLP) and conducting proactive crack sealing prior to Kiremt, RoadEye™ saves an estimated 15 to 30 Million ETB per year for Debre Berhan Municipality.", "Massive Municipal Budget Savings:"))
    body.append(bullet("Eliminates millions of USD in imports of proprietary laser equipment and foreign inspection consultants.", "Foreign Currency Preservation:"))
    body.append(bullet("Eliminating severe potholes and providing in-cabin collision alerts significantly reduces rollover collisions, sudden lane-swerving crashes, and vehicle damage on Route A2.", "Saving Human Lives:"))
    body.append(bullet("Generates high-skill technology employment in AI annotation, GIS data analysis, sensor installation, and municipal infrastructure management in Debre Berhan.", "Youth Employment & Tech Innovation:"))
    body.append(bullet("Preventive localized patching uses only 5% of the carbon-intensive hot bitumen and aggregate required for full-depth reconstruction.", "Environmental Sustainability:"))

    # Section 8
    body.append(h1("8. Team Members and Contact Information"))
    body.append(table(
        ["Parameter", "Details"],
        [
            ["Lead Innovator", "Haile Sintayehu"],
            ["Role", "Lead AI & Computer Vision Developer / System Architect"],
            ["Institution / College", "Debre Berhan University – College of Computing"],
            ["Email Address", "hailesintayehu113@gmail.com"],
            ["Phone Contact", "0921746222 / 0912485346"],
            ["GitHub Portfolio", "github.com/haile6 / intelligent-road-monitoring-system"],
            ["Technical Advisory", "Debre Berhan University – Faculty of Computing & Faculty of Civil Engineering Mentors"]
        ]
    ))
    body.append(p("", space_after=150))
    body.append(p("Declaration: We submit this business proposal to the Debre Berhan University Center for Employability and Entrepreneurship Development and the Entrepreneurship Development Institute (EDI) for the Global Entrepreneurship Week 2026 Business Idea Competition. We verify that RoadEye™ is an original, technically validated innovation developed at Debre Berhan University ready for immediate pilot deployment and commercial incubation.", italic=True))
    body.append(p("", space_after=80))
    body.append(p("Applicant Signature: Haile Sintayehu                         Date: November 16, 2026", bold=True))

    # Assemble document.xml
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {"".join(body)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>
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

    print(f"Generated DOCX: {docx_path} ({os.path.getsize(docx_path)} bytes)")

if __name__ == "__main__":
    make_docx()

