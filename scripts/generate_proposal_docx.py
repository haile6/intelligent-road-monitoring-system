"""
Generate Word-Compatible Document (.doc) for Debre Berhan University & EDI
Business Idea Competition 2026 - Concise, Driver-Alert Focused Version.
"""

import os

HTML_DOC_CONTENT = """<!DOCTYPE html>
<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head>
<meta charset="utf-8">
<title>DBU EDI Business Idea Proposal - RoadEye</title>
<!--[if gte mso 9]>
<xml>
<w:WordDocument>
<w:View>Print</w:View>
<w:Zoom>100</w:Zoom>
<w:DoNotOptimizeForBrowser/>
</w:WordDocument>
</xml>
<![endif]-->
<style>
@page {
    size: A4 portrait;
    margin: 1.8cm 1.8cm 1.8cm 1.8cm;
}
body {
    font-family: 'Segoe UI', Calibri, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1f2937;
    background-color: #ffffff;
}
h1 {
    font-size: 18pt;
    font-weight: 700;
    color: #1e3a8a;
    text-align: center;
    margin-top: 10pt;
    margin-bottom: 4pt;
}
h2 {
    font-size: 13pt;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 3pt;
    margin-top: 14pt;
    margin-bottom: 6pt;
}
h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #0369a1;
    margin-top: 10pt;
    margin-bottom: 3pt;
}
p {
    margin-bottom: 5pt;
    text-align: justify;
}
ul {
    margin-top: 3pt;
    margin-bottom: 6pt;
    padding-left: 18pt;
}
li {
    margin-bottom: 3pt;
}
.header-box {
    background-color: #f8fafc;
    border: 2px solid #cbd5e1;
    border-radius: 6px;
    padding: 10pt;
    text-align: center;
    margin-bottom: 10pt;
}
.institution-title {
    font-size: 14pt;
    font-weight: bold;
    color: #1e3a8a;
    text-transform: uppercase;
}
.center-title {
    font-size: 11pt;
    font-weight: 600;
    color: #334155;
}
.competition-badge {
    display: inline-block;
    background-color: #1e40af;
    color: #ffffff;
    font-weight: bold;
    font-size: 10.5pt;
    padding: 3pt 10pt;
    border-radius: 4px;
    margin-top: 5pt;
}
table.meta-table, table.data-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 6pt;
    margin-bottom: 10pt;
    font-size: 9.5pt;
}
table.meta-table td {
    padding: 3pt 6pt;
    border: 1px solid #cbd5e1;
}
.meta-label {
    font-weight: bold;
    color: #1e293b;
    width: 25%;
    background-color: #f1f5f9 !important;
}
table.data-table th {
    background-color: #1e3a8a;
    color: #ffffff;
    font-weight: bold;
    text-align: left;
    padding: 5pt 6pt;
    border: 1px solid #1e3a8a;
}
table.data-table td {
    padding: 4pt 6pt;
    border: 1px solid #cbd5e1;
}
table.data-table tr:nth-child(even) {
    background-color: #f8fafc;
}
.highlight-box {
    background-color: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 6pt 10pt;
    margin: 6pt 0;
    font-size: 10pt;
}
.signature-section {
    margin-top: 18pt;
    border-top: 1px solid #cbd5e1;
    padding-top: 8pt;
}
</style>
</head>
<body>

<div class="header-box">
    <div class="institution-title">Debre Berhan University</div>
    <div class="center-title">Center for Employability and Entrepreneurship Development</div>
    <div style="font-size: 9.5pt; color: #64748b;">In Collaboration with Entrepreneurship Development Institute (EDI)</div>
    <div class="competition-badge">BUSINESS IDEA COMPETITION PROPOSAL • GLOBAL ENTREPRENEURSHIP WEEK 2026</div>
</div>

<table class="meta-table">
    <tr>
        <td class="meta-label">Submission Deadline:</td>
        <td><strong>November 16, 2026 – 5:00 PM</strong></td>
        <td class="meta-label">Submission Format:</td>
        <td>Word / PDF Document</td>
    </tr>
    <tr>
        <td class="meta-label">Submission Email:</td>
        <td><strong>deredbu2003@gmail.com</strong></td>
        <td class="meta-label">Office Location:</td>
        <td>Center for Employability / SE-02, F2</td>
    </tr>
    <tr>
        <td class="meta-label">Lead Innovator:</td>
        <td><strong>Haile Sintayehu</strong></td>
        <td class="meta-label">Contact / Phone:</td>
        <td><strong>0921746222 / 0912485346</strong></td>
    </tr>
    <tr>
        <td class="meta-label">College / Dept:</td>
        <td>College of Computing</td>
        <td class="meta-label">Prototype Status:</td>
        <td><strong>114 Automated Tests Passing (100%)</strong></td>
    </tr>
</table>

<div style="text-align: center; margin-top: 6pt; margin-bottom: 10pt;">
    <h1 style="color: #1e3a8a; margin-bottom: 2pt;">RoadEye™</h1>
    <div style="font-size: 12pt; font-weight: 600; color: #0284c7;">Dual-Action Platform: Real-Time In-Cabin Driver Voice Warning & Municipal Asset Management</div>
    <div style="font-size: 9pt; color: #64748b;">Repository: <a href="https://github.com/haile6/intelligent-road-monitoring-system">github.com/haile6/intelligent-road-monitoring-system</a></div>
</div>

<h2>1. Business Idea Title and Description</h2>
<p><strong>Title:</strong> RoadEye™: Edge-AI In-Cabin Driver Safety Voice Alarm & Municipal Asset Management Platform</p>
<p><strong>Overview:</strong> <strong>RoadEye™</strong> is an innovative GovTech and Mobility AI platform engineered at Debre Berhan University. It converts affordable dashcams and smartphones into a <strong>dual-action system</strong>:</p>
<ul>
    <li><strong>1. For Drivers (Active In-Cabin ADAS Safety):</strong> Continuously monitors the forward driving corridor, calculates closing Time-To-Collision (TTC &le; 1.8s), and sounds <strong>real-time voice alarms and visual HUD alerts in both Amharic (አማርኛ) and English</strong> (e.g., <em>"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"</em> / <em>"Warning! Pothole Ahead, Reduce Speed"</em>). This prevents violent swerving, rollovers, and tire blowouts.</li>
    <li><strong>2. For Municipalities (GovTech Asset Management):</strong> Automatically detects pavement distresses, calculates 3D cavity depth and required asphalt repair mass (kg), projects defects onto an interactive GIS map, and enforces contractor warranties under FIDIC Clause 11 (Defect Liability Period).</li>
</ul>
<p>By identifying road defects before the Ethiopian monsoon (<strong>Kiremt</strong>, where damage accelerates by <strong>4.2&times;</strong>), RoadEye™ cuts municipal repair costs by up to <strong>75%</strong> while protecting drivers' lives daily.</p>

<h2>2. Problem Statement and Market Need</h2>
<ul>
    <li><strong>1. Lethal Driver Collisions from Sudden Potholes:</strong> Drivers traveling at high speeds encounter unexpected potholes, causing violent swerving into oncoming traffic, rollovers, or tire blowouts. Without automated in-cabin voice warnings, drivers cannot react safely at night or in rain.</li>
    <li><strong>2. Inefficient & Dangerous Manual Inspections:</strong> Technicians walk along high-speed corridors with tape measures and clipboards. It is lethally dangerous, covers under 5% of the network annually, and takes months to report.</li>
    <li><strong>3. Severe Monsoon (Kiremt) Pavement Destruction:</strong> Heavy rainfall and freight truck axle loads create hydrodynamic pore-water pumping, accelerating pavement breakdown by <strong>4.2&times;</strong> during Kiremt.</li>
    <li><strong>4. Millions Lost from Unenforced Contractor Warranties:</strong> When newly built roads fail during the 1–2 year warranty period (FIDIC Clause 11 DLP), municipalities lack digital evidence, forcing taxpayers to pay tens of millions of ETB that contractors should fix for free.</li>
    <li><strong>5. Foreign Currency (Forex) Crisis:</strong> Foreign laser inspection vans (ROMDAS) cost <strong>$250,000–$500,000 USD</strong>—completely unaffordable for Ethiopian municipal budgets.</li>
</ul>

<h2>3. Proposed Product and Technical Solution</h2>
<div class="highlight-box">
    <strong>Key Capabilities:</strong><br/>
    • <strong>Active Driver Voice Announcement & ADAS Alarm:</strong> Monitors forward travel lanes at 30+ FPS and issues dual-language auditory voice warnings (<em>"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"</em>) and flashing HUD displays when TTC &le; 1.8 seconds.<br/>
    • <strong>Monocular 3D Cavity & Asphalt Quantification:</strong> Directly computes depth (cm), volume (Liters), and exact commercial asphalt mix mass (in kg and 25kg bags) compliant with ERA specifications.<br/>
    • <strong>Multi-Vehicle Fleet Fusion:</strong> Merges duplicate detections from multiple vehicles within 8 meters, achieving 99.9% statistical certainty.<br/>
    • <strong>Contractor Warranty Engine (FIDIC Clause 11):</strong> Cross-references defect coordinates with contractor warranty zones and unit rates (28,000 ETB/pothole, 18,000 ETB/crack) to produce formal legal rectification demand notices.<br/>
    • <strong>Rainy Season Cost-of-Inaction (COI) ROI Engine:</strong> Proves that spending 2,850 ETB/m² on proactive crack sealing today saves 7,800 ETB/m² in post-monsoon reconstruction (<strong>8.5&times;–11.2&times; ROI</strong>).
</div>

<h2>4. Target Customers, Market Potential & Key Drivers</h2>
<h3>4.1 Target Customer Segments</h3>
<ul>
    <li><strong>Municipalities (GovTech B2G):</strong> Debre Berhan City Administration, Zonal & Regional Urban Infrastructure Bureaus.</li>
    <li><strong>National Highway Authorities:</strong> Ethiopian Roads Administration (ERA) and Regional Road Bureaus.</li>
    <li><strong>Road Construction Contractors (B2B):</strong> Defence Construction, Sur Construction, Midroc, Yencomad (for quality audits & warranty defense).</li>
    <li><strong>Commercial Transport Fleets:</strong> Freight trucks on Route A2, bus fleets, and logistics companies requiring in-cabin driver alerts.</li>
</ul>

<h3>4.2 Market Potential & Key Market Drivers</h3>
<ul>
    <li><strong>TAM / SAM / SOM:</strong> TAM: 160,000+ km of roads (> 25 Billion ETB national maintenance context); SAM: Amhara Region & Central Corridors (180 Million ETB/year); SOM: Debre Berhan network & A2 corridor (12–18 Million ETB/year).</li>
    <li><strong>1. Federal Road Fund Digital Mandate:</strong> MUI and ERA require digitized Pavement Condition Index (PCI) logs before approving municipal road budgets.</li>
    <li><strong>2. Strict Anti-Corruption & Public Procurement Audits (OFAG):</strong> The Federal Auditor General holds municipal officials personally liable if they pay for repairs that contractors should fix for free under warranty.</li>
    <li><strong>3. Urban Corridor Development Boom:</strong> Rapid asphalt expansion in Debre Berhan and regional cities creates billions of ETB in new road assets currently in their 1–2 year warranty periods.</li>
    <li><strong>4. Forex Crunch & Import Substitution Policy:</strong> Foreign exchange shortages make imported laser survey vans ($500k USD) unaffordable, driving urgent demand for 100% locally built GovTech solutions.</li>
    <li><strong>5. Construction Material Inflation (>250%):</strong> Rebuilding roads is unaffordable; cities must prioritize proactive, early-stage crack sealing.</li>
</ul>

<h2>5. Innovation and Competitive Advantage</h2>
<table class="data-table">
    <thead>
        <tr>
            <th>Feature</th>
            <th>Manual Survey</th>
            <th>Imported Laser Vans</th>
            <th>RoadEye™ (Our Innovation)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Hardware Cost</strong></td>
            <td>Low</td>
            <td><strong>$250,000 – $500,000 USD</strong></td>
            <td><strong>&lt; $350 USD</strong> (Standard dashcam / phone)</td>
        </tr>
        <tr>
            <td><strong>Forex Dependency</strong></td>
            <td>None</td>
            <td><strong>100% Forex Dependent</strong></td>
            <td><strong>0% Forex</strong> (100% locally built & hosted)</td>
        </tr>
        <tr>
            <td><strong>Driver Voice Alarm</strong></td>
            <td>None</td>
            <td>None</td>
            <td><strong>Real-Time Dual-Language (አማርኛ / English)</strong></td>
        </tr>
        <tr>
            <td><strong>Survey Speed</strong></td>
            <td>3 – 5 km/day</td>
            <td>40 – 60 km/day</td>
            <td><strong>100 – 300+ km/day</strong></td>
        </tr>
        <tr>
            <td><strong>Worker Risk</strong></td>
            <td>Extremely High Risk</td>
            <td>Low Risk</td>
            <td><strong>Zero Risk</strong> (Surveyed from vehicle cabin)</td>
        </tr>
        <tr>
            <td><strong>3D Cavity Material</strong></td>
            <td>Guesswork</td>
            <td>Requires LiDAR</td>
            <td><strong>Automated Monocular 3D (Bags & kg)</strong></td>
        </tr>
        <tr>
            <td><strong>Contractor DLP</strong></td>
            <td>None</td>
            <td>Manual / None</td>
            <td><strong>Automated FIDIC Clause 11 Geofencing</strong></td>
        </tr>
        <tr>
            <td><strong>Prototype Readiness</strong></td>
            <td>Legacy</td>
            <td>Commercial</td>
            <td><strong>Working Prototype (114 Passing Tests)</strong></td>
        </tr>
    </tbody>
</table>

<h2>6. Business Model and Commercialization Strategy</h2>
<ul>
    <li><strong>1. Municipal SaaS Subscriptions:</strong> Basic (250,000 ETB), Pro / Debre Berhan (650,000 ETB), Enterprise Metro (1,800,000 ETB).</li>
    <li><strong>2. Pavement Inspection-as-a-Service (PIaaS):</strong> Turnkey 3D surveying at 1,200–1,800 ETB per kilometer.</li>
    <li><strong>3. Contractor DLP Compliance Audits:</strong> Pre-handover audit reports at 45,000–120,000 ETB per project.</li>
    <li><strong>4. Fleet Safety & Telemetry Feeds:</strong> Monthly subscription for in-cabin driver alerts and route hazard data.</li>
</ul>

<h3>3-Year Financial Forecast:</h3>
<table class="data-table">
    <thead>
        <tr>
            <th>Year</th>
            <th>Strategic Milestone</th>
            <th>Clients / Projects</th>
            <th>Projected Revenue</th>
            <th>Net Margin</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>2027</strong></td>
            <td>Debre Berhan Pilot & Fleet Launch</td>
            <td>1 Municipality + 4 Contractors</td>
            <td><strong>2,450,000 ETB</strong></td>
            <td><strong>42%</strong></td>
        </tr>
        <tr>
            <td><strong>2028</strong></td>
            <td>Expansion to 5 Amhara Cities + ERA</td>
            <td>6 Municipalities + 12 Contractors</td>
            <td><strong>8,900,000 ETB</strong></td>
            <td><strong>56%</strong></td>
        </tr>
        <tr>
            <td><strong>2029</strong></td>
            <td>National Corridor Expansion</td>
            <td>18 Municipalities + 35 Contractors</td>
            <td><strong>26,500,000 ETB</strong></td>
            <td><strong>68%</strong></td>
        </tr>
    </tbody>
</table>

<h2>7. Expected Economic, Social, and Safety Impact</h2>
<ul>
    <li><strong>Saving Driver Lives & Preventing Crashes:</strong> Real-time voice announcements (<em>"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"</em>) prevent violent swerving, rollovers, and multi-vehicle highway collisions.</li>
    <li><strong>Massive Municipal Savings:</strong> Proactive crack sealing and contractor warranty enforcement save <strong>15–30 Million ETB</strong> annually for Debre Berhan City Administration.</li>
    <li><strong>Preserving Foreign Currency:</strong> Replaces scarce foreign currency imports with locally engineered software and standard commercial hardware.</li>
    <li><strong>Youth Employment & Tech Innovation:</strong> Creates high-skill jobs in AI engineering, computer vision, GIS analysis, and municipal infrastructure management in Debre Berhan.</li>
</ul>

<h2>8. Team Information and Contact Details</h2>
<table class="meta-table">
    <tr>
        <td class="meta-label">Lead Innovator:</td>
        <td><strong>Haile Sintayehu</strong></td>
        <td class="meta-label">Position:</td>
        <td>Lead AI & Computer Vision Developer / System Architect</td>
    </tr>
    <tr>
        <td class="meta-label">Institution:</td>
        <td>Debre Berhan University – College of Computing</td>
        <td class="meta-label">Email:</td>
        <td><strong>hailesintayehu113@gmail.com</strong></td>
    </tr>
    <tr>
        <td class="meta-label">Contact Numbers:</td>
        <td><strong>0921746222 / 0912485346</strong></td>
        <td class="meta-label">Repository:</td>
        <td><a href="https://github.com/haile6/intelligent-road-monitoring-system">haile6/intelligent-road-monitoring-system</a></td>
    </tr>
</table>

<div class="signature-section">
    <p><em>Declaration: I submit this business proposal to the Debre Berhan University Center for Employability and Entrepreneurship Development and EDI for the Global Entrepreneurship Week 2026 Business Idea Competition. RoadEye™ is an original innovation developed at Debre Berhan University, validated by 114 passing unit tests, ready for pilot deployment and commercial incubation.</em></p>
    <br/>
    <p><strong>Applicant Signature:</strong> <em>Haile Sintayehu</em> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Date:</strong> November 16, 2026</p>
</div>

</body>
</html>
"""

def generate_docs():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    doc_path = os.path.join(output_dir, "DBU_EDI_Business_Idea_Proposal_RoadEye.doc")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(HTML_DOC_CONTENT)
    print(f"Successfully generated clean DOC: {doc_path} ({os.path.getsize(doc_path)} bytes)")

if __name__ == "__main__":
    generate_docs()
