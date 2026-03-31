"""Generate a one-page PDF summary for a doctor visit."""

from fpdf import FPDF
from datetime import date
import io


class _ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, "GlycoCast - Diabetes Risk Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f"Generated {date.today().strftime('%B %d, %Y')}  |  For discussion with your healthcare provider", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, "This is an educational estimate, not a medical diagnosis. Please consult a healthcare provider.", align="C")


def generate_report_pdf(data: dict) -> bytes:
    """Build a single-page PDF and return raw bytes."""
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    risk_pct = data.get("risk_pct", 0)
    risk_label = data.get("risk_label", "")

    # ── Risk Score Hero ──
    pdf.set_font("Helvetica", "B", 36)
    if risk_pct >= 60:
        pdf.set_text_color(214, 46, 48)
    elif risk_pct >= 30:
        pdf.set_text_color(184, 105, 8)
    else:
        pdf.set_text_color(43, 138, 80)
    pdf.cell(0, 16, f"{risk_pct}%  Estimated Diabetes Risk", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, _sanitize(f"Classification: {risk_label}"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Key Metrics Table ──
    _section(pdf, "Key Metrics")
    metrics = [
        ("Fasting Glucose", f"{data.get('glucose_display', 'N/A')} mg/dL", "estimated" if data.get("glucose_estimated") else "user-supplied"),
        ("BMI", f"{data.get('bmi', 'N/A')}", "user-supplied"),
        ("Blood Pressure (dia)", f"{data.get('bp', 'N/A')} mmHg", "user-supplied"),
        ("Waist Circumference", f"{data.get('waist', 'N/A')} cm", "user-supplied"),
        ("Insulin Resistance Tier", data.get("ir_tier", "N/A"), f"HOMA-IR ~{data.get('homa_range', 'N/A')}"),
        ("FINDRISC Score", f"{data.get('findrisc', 'N/A')} / 26", data.get("findrisc_label", "")),
    ]
    col_w = [60, 50, 70]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(["Metric", "Value", "Note"], col_w):
        pdf.cell(w, 6, h, border="B")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for label, value, note in metrics:
        pdf.cell(col_w[0], 5.5, _sanitize(label))
        pdf.cell(col_w[1], 5.5, _sanitize(str(value)))
        pdf.cell(col_w[2], 5.5, _sanitize(str(note)))
        pdf.ln()
    pdf.ln(3)

    # ── Top Risk Drivers ──
    risk_bars = data.get("risk_factor_bars", [])[:5]
    if risk_bars:
        _section(pdf, "Top Risk Drivers")
        pdf.set_font("Helvetica", "", 9)
        for f in risk_bars:
            pdf.cell(0, 5, f"  -  {f['name']} - {f['note']}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Action Plan ──
    actions = data.get("actions", [])[:5]
    if actions:
        _section(pdf, "Recommended Actions")
        pdf.set_font("Helvetica", "", 9)
        for a in actions:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5.5, _sanitize(f"  [{a['badge']}]  {a['title']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 4.5, _sanitize(f"      {a['delta']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── Questions for Doctor ──
    ask = data.get("ask_doctor", [])
    if ask:
        _section(pdf, "Questions to Ask Your Doctor")
        pdf.set_font("Helvetica", "", 9)
        for q in ask:
            pdf.cell(0, 5, f"  -  {_sanitize(q)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Suggested Lab Tests ──
    labs = data.get("lab_tests", [])
    if labs:
        _section(pdf, "Suggested Lab Tests")
        pdf.set_font("Helvetica", "", 9)
        for t in labs:
            pdf.cell(0, 5, f"  -  {_sanitize(t)}", new_x="LMARGIN", new_y="NEXT")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _sanitize(text: str) -> str:
    """Replace characters outside latin-1 with safe ASCII equivalents."""
    replacements = {"\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
                    "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2026": "...",
                    "\u00b5": "u"}
    for ch, repl in replacements.items():
        text = text.replace(ch, repl)
    return text.encode("latin-1", "replace").decode("latin-1")


def _section(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(232, 244, 253)
    pdf.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
