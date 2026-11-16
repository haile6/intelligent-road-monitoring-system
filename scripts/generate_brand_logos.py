"""
Generate Premium Vector SVG & PNG Brand Logos for RoadEye™.
Local generation - DO NOT PUSH TO GITHUB.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import math

SVG_ICON = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <!-- Background Gradient -->
    <radialGradient id="bgGrad" cx="50%" cy="50%" r="55%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="60%" stop-color="#090E17"/>
      <stop offset="100%" stop-color="#04070D"/>
    </radialGradient>

    <!-- Cyan Glow -->
    <linearGradient id="cyanGlow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="50%" stop-color="#00D2FF"/>
      <stop offset="100%" stop-color="#0284C7"/>
    </linearGradient>

    <!-- Emerald Accent -->
    <linearGradient id="emeraldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#34D399"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>

    <!-- Road Surface Gradient -->
    <linearGradient id="roadGrad" x1="50%" y1="100%" x2="50%" y2="0%">
      <stop offset="0%" stop-color="#1E293B"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>

    <!-- Shield Border Gradient -->
    <linearGradient id="shieldBorder" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.8"/>
      <stop offset="50%" stop-color="#1E293B" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#00D2FF" stop-opacity="0.8"/>
    </linearGradient>

    <!-- Glow Filter -->
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>

    <filter id="subtleGlow" x="-10%" y="-10%" width="120%" height="120%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
  </defs>

  <!-- Background Shield / Rounded Squircle -->
  <rect x="16" y="16" width="480" height="480" rx="96" fill="url(#bgGrad)" stroke="url(#shieldBorder)" stroke-width="4"/>

  <!-- Radar Proximity Scanning Rings -->
  <circle cx="256" cy="256" r="210" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="6 8" stroke-opacity="0.25"/>
  <circle cx="256" cy="256" r="160" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-opacity="0.35"/>
  <circle cx="256" cy="256" r="110" fill="none" stroke="#00D2FF" stroke-width="1.5" stroke-dasharray="4 6" stroke-opacity="0.45"/>

  <!-- Radar 4-Axis Reticle Crosshairs -->
  <line x1="256" y1="56" x2="256" y2="86" stroke="#38BDF8" stroke-width="2.5" stroke-opacity="0.6"/>
  <line x1="256" y1="426" x2="256" y2="456" stroke="#38BDF8" stroke-width="2.5" stroke-opacity="0.6"/>
  <line x1="56" y1="256" x2="86" y2="256" stroke="#38BDF8" stroke-width="2.5" stroke-opacity="0.6"/>
  <line x1="426" y1="256" x2="456" y2="256" stroke="#38BDF8" stroke-width="2.5" stroke-opacity="0.6"/>

  <!-- Cybernetic Eye Outer Arcs (Aperture) -->
  <!-- Upper Eyelid Arc -->
  <path d="M 96 256 C 146 156, 366 156, 416 256" fill="none" stroke="url(#cyanGlow)" stroke-width="8" stroke-linecap="round" filter="url(#glow)"/>
  <!-- Lower Eyelid Arc -->
  <path d="M 96 256 C 146 356, 366 356, 416 256" fill="none" stroke="url(#cyanGlow)" stroke-width="8" stroke-linecap="round" filter="url(#glow)"/>

  <!-- Receding Perspective Highway / Corridor -->
  <!-- Road Polygon -->
  <polygon points="256,220 220,380 292,380" fill="url(#roadGrad)" stroke="#334155" stroke-width="2"/>
  
  <!-- Road Outer Curbs (Green / Safety Glow) -->
  <line x1="256" y1="220" x2="220" y2="380" stroke="url(#emeraldGrad)" stroke-width="4" stroke-linecap="round" filter="url(#subtleGlow)"/>
  <line x1="256" y1="220" x2="292" y2="380" stroke="url(#emeraldGrad)" stroke-width="4" stroke-linecap="round" filter="url(#subtleGlow)"/>

  <!-- Highway Center Dashed Stripe (Glowing Yellow / White) -->
  <line x1="256" y1="230" x2="256" y2="255" stroke="#FBBF24" stroke-width="3" stroke-linecap="round"/>
  <line x1="256" y1="270" x2="256" y2="305" stroke="#FBBF24" stroke-width="3.5" stroke-linecap="round"/>
  <line x1="256" y1="325" x2="256" y2="370" stroke="#FBBF24" stroke-width="4.5" stroke-linecap="round"/>

  <!-- Central Camera Aperture / Iris (AI Core) -->
  <circle cx="256" cy="220" r="50" fill="none" stroke="url(#cyanGlow)" stroke-width="5" filter="url(#glow)"/>
  <circle cx="256" cy="220" r="30" fill="#090E17" stroke="#38BDF8" stroke-width="3"/>
  <circle cx="256" cy="220" r="14" fill="#00D2FF" filter="url(#glow)"/>
  <circle cx="256" cy="220" r="5" fill="#FFFFFF"/>

  <!-- Forward Sonar / Laser Pulse Beam -->
  <polygon points="256,170 236,120 276,120" fill="url(#cyanGlow)" opacity="0.35"/>
  <circle cx="256" cy="120" r="6" fill="#38BDF8" filter="url(#glow)"/>
</svg>'''

SVG_HORIZONTAL = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 300" width="1000" height="300">
  <defs>
    <!-- Background Gradient -->
    <linearGradient id="bgHGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0B132B"/>
      <stop offset="50%" stop-color="#080D1A"/>
      <stop offset="100%" stop-color="#03060B"/>
    </linearGradient>

    <!-- Cyan Glow -->
    <linearGradient id="cyanH" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="60%" stop-color="#00D2FF"/>
      <stop offset="100%" stop-color="#0284C7"/>
    </linearGradient>

    <!-- Emerald Accent -->
    <linearGradient id="emeraldH" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#34D399"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>

    <filter id="glowH" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
  </defs>

  <!-- Background Container -->
  <rect x="0" y="0" width="1000" height="300" rx="32" fill="url(#bgHGrad)"/>
  <rect x="0" y="0" width="1000" height="300" rx="32" fill="none" stroke="#1E293B" stroke-width="2"/>

  <!-- Left Icon Group (Transformed to fit left side x=150, y=150) -->
  <g transform="translate(40, 25) scale(0.48)">
    <!-- Radar Rings -->
    <circle cx="256" cy="256" r="210" fill="none" stroke="#38BDF8" stroke-width="2" stroke-dasharray="8 10" stroke-opacity="0.3"/>
    <circle cx="256" cy="256" r="150" fill="none" stroke="#38BDF8" stroke-width="2" stroke-opacity="0.4"/>
    <circle cx="256" cy="256" r="90" fill="none" stroke="#00D2FF" stroke-width="2" stroke-dasharray="6 6" stroke-opacity="0.5"/>

    <!-- Eye Outlines -->
    <path d="M 80 256 C 130 140, 382 140, 432 256" fill="none" stroke="url(#cyanH)" stroke-width="12" stroke-linecap="round" filter="url(#glowH)"/>
    <path d="M 80 256 C 130 372, 382 372, 432 256" fill="none" stroke="url(#cyanH)" stroke-width="12" stroke-linecap="round" filter="url(#glowH)"/>

    <!-- Highway -->
    <polygon points="256,220 210,400 302,400" fill="#0F172A" stroke="#334155" stroke-width="2"/>
    <line x1="256" y1="220" x2="210" y2="400" stroke="url(#emeraldH)" stroke-width="6" stroke-linecap="round" filter="url(#glowH)"/>
    <line x1="256" y1="220" x2="302" y2="400" stroke="url(#emeraldH)" stroke-width="6" stroke-linecap="round" filter="url(#glowH)"/>
    <line x1="256" y1="235" x2="256" y2="265" stroke="#FBBF24" stroke-width="4" stroke-linecap="round"/>
    <line x1="256" y1="285" x2="256" y2="325" stroke="#FBBF24" stroke-width="5" stroke-linecap="round"/>
    <line x1="256" y1="345" x2="256" y2="390" stroke="#FBBF24" stroke-width="6" stroke-linecap="round"/>

    <!-- Pupil Aperture -->
    <circle cx="256" cy="220" r="54" fill="none" stroke="url(#cyanH)" stroke-width="7" filter="url(#glowH)"/>
    <circle cx="256" cy="220" r="32" fill="#080D1A" stroke="#38BDF8" stroke-width="4"/>
    <circle cx="256" cy="220" r="15" fill="#00D2FF" filter="url(#glowH)"/>
    <circle cx="256" cy="220" r="6" fill="#FFFFFF"/>
  </g>

  <!-- Typography Right Side -->
  <g transform="translate(320, 0)">
    <!-- Brand Title: Road (White) + Eye (Cyan) -->
    <text x="0" y="145" font-family="'Segoe UI', 'Montserrat', 'Helvetica', sans-serif" font-size="78" font-weight="800" fill="#FFFFFF" letter-spacing="1">
      Road<tspan fill="url(#cyanH)" filter="url(#glowH)">Eye</tspan><tspan font-size="34" font-weight="700" fill="#38BDF8" dy="-32">™</tspan>
    </text>

    <!-- Subtitle / Tagline -->
    <text x="5" y="195" font-family="'Segoe UI', 'Arial', sans-serif" font-size="20" font-weight="700" fill="#94A3B8" letter-spacing="3.5">
      AI ROAD INFRASTRUCTURE INTELLIGENCE
    </text>

    <!-- Secondary Capability Badges -->
    <g transform="translate(5, 222)">
      <!-- Badge 1: 3D CV -->
      <rect x="0" y="0" width="150" height="28" rx="6" fill="#1E293B" stroke="#38BDF8" stroke-width="1" stroke-opacity="0.4"/>
      <text x="75" y="18" font-family="'Segoe UI', sans-serif" font-size="11.5" font-weight="600" fill="#38BDF8" text-anchor="middle">3D COMPUTER VISION</text>

      <!-- Badge 2: ADAS RADAR -->
      <rect x="160" y="0" width="145" height="28" rx="6" fill="#1E293B" stroke="#34D399" stroke-width="1" stroke-opacity="0.4"/>
      <text x="232" y="18" font-family="'Segoe UI', sans-serif" font-size="11.5" font-weight="600" fill="#34D399" text-anchor="middle">ADAS VOICE RADAR</text>

      <!-- Badge 3: FIDIC DLP -->
      <rect x="315" y="0" width="165" height="28" rx="6" fill="#1E293B" stroke="#FBBF24" stroke-width="1" stroke-opacity="0.4"/>
      <text x="397" y="18" font-family="'Segoe UI', sans-serif" font-size="11.5" font-weight="600" fill="#FBBF24" text-anchor="middle">MUNICIPAL ASSET SAAS</text>
    </g>
  </g>
</svg>'''

def generate_svgs():
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets")
    static_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "static", "icons")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)

    icon_svg_path = os.path.join(assets_dir, "roadeye_logo_icon.svg")
    with open(icon_svg_path, "w", encoding="utf-8") as f:
        f.write(SVG_ICON)
    print(f"Saved: {icon_svg_path}")

    horiz_svg_path = os.path.join(assets_dir, "roadeye_logo_horizontal.svg")
    with open(horiz_svg_path, "w", encoding="utf-8") as f:
        f.write(SVG_HORIZONTAL)
    print(f"Saved: {horiz_svg_path}")

    # Also save in web/static/icons/ for dashboard branding
    web_icon_path = os.path.join(static_img_dir, "roadeye_logo.svg")
    with open(web_icon_path, "w", encoding="utf-8") as f:
        f.write(SVG_ICON)
    print(f"Saved: {web_icon_path}")

def generate_highres_png():
    """Generate high-res PNG using Pillow for immediate insertion into Word / Presentations."""
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets")
    png_path = os.path.join(assets_dir, "roadeye_logo_icon.png")

    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2

    # Draw Dark Background Squircle
    margin = 32
    draw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=180, fill=(11, 19, 43, 255), outline=(56, 189, 248, 200), width=8)

    # Radar concentric circles
    for r, width in [(420, 2), (320, 3), (220, 3)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(56, 189, 248, 70), width=width)

    # Radar Crosshair Ticks
    draw.line([cx, 80, cx, 140], fill=(56, 189, 248, 180), width=5)
    draw.line([cx, size - 140, cx, size - 80], fill=(56, 189, 248, 180), width=5)
    draw.line([80, cy, 140, cy], fill=(56, 189, 248, 180), width=5)
    draw.line([size - 140, cy, size - 80, cy], fill=(56, 189, 248, 180), width=5)

    # Upper and Lower Eye Arc (Approximated with smooth chords/curves)
    # Upper arc
    draw.arc([160, 160, 864, 864], start=210, end=330, fill=(0, 210, 255, 255), width=16)
    # Lower arc
    draw.arc([160, 160, 864, 864], start=30, end=150, fill=(0, 210, 255, 255), width=16)

    # Highway Corridor Polygon (Receding road)
    road_poly = [(cx, 440), (cx - 80, 760), (cx + 80, 760)]
    draw.polygon(road_poly, fill=(20, 30, 50, 255), outline=(51, 65, 85, 255))

    # Road Curbs (Emerald Green)
    draw.line([(cx, 440), (cx - 80, 760)], fill=(52, 211, 153, 255), width=8)
    draw.line([(cx, 440), (cx + 80, 760)], fill=(52, 211, 153, 255), width=8)

    # Road Center Dashes (Yellow)
    draw.line([(cx, 470), (cx, 520)], fill=(251, 191, 36, 255), width=6)
    draw.line([(cx, 550), (cx, 620)], fill=(251, 191, 36, 255), width=8)
    draw.line([(cx, 660), (cx, 750)], fill=(251, 191, 36, 255), width=10)

    # Central Aperture / Eye Iris
    draw.ellipse([cx - 100, 440 - 100, cx + 100, 440 + 100], outline=(0, 210, 255, 255), width=10)
    draw.ellipse([cx - 60, 440 - 60, cx + 60, 440 + 60], fill=(8, 13, 26, 255), outline=(56, 189, 248, 255), width=6)
    draw.ellipse([cx - 30, 440 - 30, cx + 30, 440 + 30], fill=(0, 210, 255, 255))
    draw.ellipse([cx - 10, 440 - 10, cx + 10, 440 + 10], fill=(255, 255, 255, 255))

    # Top Sonar pulse
    draw.polygon([(cx, 320), (cx - 40, 240), (cx + 40, 240)], fill=(0, 210, 255, 90))
    draw.ellipse([cx - 12, 240 - 12, cx + 12, 240 + 12], fill=(56, 189, 248, 255))

    img.save(png_path, "PNG")
    print(f"Generated High-Res PNG: {png_path} ({os.path.getsize(png_path)} bytes)")

if __name__ == "__main__":
    generate_svgs()
    generate_highres_png()
