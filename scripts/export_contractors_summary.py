#!/usr/bin/env python3
"""
Contractor Warranty & Performance Audit Exporter CLI.
Extracts nationwide Ethiopian road contractor telemetry, Defect Liability Period (DLP)
warranty status, and financial liability estimates into Markdown, CSV, or JSON reports.
"""
import os
import sys
import argparse
import json
import csv
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'buffer'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import database.db_manager as db_manager


def export_audit_report(output_path=None, export_format="markdown"):
    """Generate and write contractor audit report."""
    analytics = db_manager.get_contractor_analytics()
    contractors = analytics.get("contractors", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if export_format == "json":
        report_data = {
            "title": "RoadEye Ethiopian Road Contractors & DLP Audit Report",
            "generated_at": now_str,
            "metrics": {
                "total_projects": analytics.get("total_contractors", 0),
                "total_corridor_km": analytics.get("total_roads_km", 0.0),
                "active_warranties": analytics.get("active_warranties", 0),
                "defects_under_warranty": analytics.get("total_defects_in_warranty", 0),
                "estimated_contractor_savings_etb": analytics.get("estimated_contractor_savings_etb", 0)
            },
            "contractors": contractors
        }
        content = json.dumps(report_data, indent=2, ensure_ascii=False)

    elif export_format == "csv":
        keys = [
            "id", "segment_code", "road_name", "contractor_name", "consultant_name",
            "length_km", "pavement_type", "handover_date", "warranty_expiry_date",
            "warranty_years", "is_under_warranty", "liability_party", "quality_grade",
            "quality_score", "incident_count", "repair_cost_etb"
        ]
        import io
        output_buffer = io.StringIO()
        writer = csv.DictWriter(output_buffer, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for c in contractors:
            writer.writerow(c)
        content = output_buffer.getvalue()

    else:  # markdown
        lines = [
            "# RoadEye™ — Ethiopian Road Contractors & DLP Warranty Audit Report",
            f"*Generated on: {now_str}*\n",
            "## 1. Executive Summary Telemetry",
            f"- **Total Monitored Corridors:** {analytics.get('total_contractors', 0)} Projects",
            f"- **Total Network Coverage:** {analytics.get('total_roads_km', 0.0):,.1f} km",
            f"- **Active Warranty Corridors:** {analytics.get('active_warranties', 0)} Projects (100% Contractor Liability)",
            f"- **Road Defects in Warranty:** {analytics.get('total_defects_in_warranty', 0)} Incidents",
            f"- **Estimated Public Treasury Savings:** {analytics.get('estimated_contractor_savings_etb', 0):,.2f} ETB\n",
            "## 2. Contractor Projects Registry & Warranty Matrix\n",
            "| Code | Road Name | Contractor | Length (km) | Handover | DLP Expiry | Warranty Status | Quality | Defects |",
            "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
        ]

        for c in contractors:
            w_status = "🟢 Active" if c.get("is_under_warranty") else "⚪ Expired"
            lines.append(
                f"| `{c.get('segment_code')}` | {c.get('road_name')} | {c.get('contractor_name')} | "
                f"{c.get('length_km')} | {c.get('handover_date')} | {c.get('warranty_expiry_date')} | "
                f"{w_status} | Grade {c.get('quality_grade')} ({c.get('quality_score')}) | {c.get('incident_count', 0)} |"
            )

        lines.append("\n---\n*Report generated automatically by RoadEye Intelligent Road Monitoring System.*")
        content = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Audit report successfully written to: {output_path} ({len(content)} bytes)")
    else:
        print(content)

    return content


def main():
    parser = argparse.ArgumentParser(description="Export Contractor Warranty Audit Report.")
    parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown",
                        help="Export format (default: markdown)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional file path to write output report")

    args = parser.parse_args()
    export_audit_report(output_path=args.output, export_format=args.format)


if __name__ == "__main__":
    main()
