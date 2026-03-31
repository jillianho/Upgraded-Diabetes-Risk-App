import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from diabetes_proxies import PatientInputs, build_feature_vector, generate_results_content
from population_stats import compute_percentiles
from pdf_export import generate_report_pdf

# ══════════════════════════════════════════════════════════════════════════════
# UNIT CONVERSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def lbs_to_kg(lbs: float) -> float:
    return lbs * 0.453592

def kg_to_lbs(kg: float) -> float:
    return kg / 0.453592

def inches_to_cm(inches: float) -> float:
    return inches * 2.54

def cm_to_inches(cm: float) -> float:
    return cm / 2.54

def mmol_to_mgdl(mmol: float) -> float:
    return mmol * 18.0

def mgdl_to_mmol(mgdl: float) -> float:
    return mgdl / 18.0

# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "risk_history.json")

def _load_history() -> list:
    if os.path.exists(_HISTORY_FILE):
        with open(_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def _save_history(history: list):
    with open(_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first!)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GlycoCast - Diabetes Risk Calculator",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state["page"] = "input"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Units & Settings
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown("**Unit preferences**")
    weight_unit = st.radio("Weight", ["kg", "lbs"], horizontal=True, key="weight_unit")
    length_unit = st.radio("Height / Waist", ["cm", "inches"], horizontal=True, key="length_unit")
    glucose_unit = st.radio("Glucose", ["mg/dL", "mmol/L"], horizontal=True, key="glucose_unit")

    st.markdown("---")
    st.markdown("**Progress history**")
    history = _load_history()
    st.caption(f"{len(history)} saved assessment(s)")
    if history and st.button("🗑️ Clear history"):
        _save_history([])
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS PAGE (keep your existing results logic)
# check
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["page"] == "results":
    st.markdown("""
    <style>
    .results-wrap { padding-top: 0.25rem; }
    .risk-hero { text-align: center; padding: 1rem 0 0.5rem; }
    .dial-wrap { display: flex; flex-direction: column; align-items: center; margin-bottom: 1rem; }
    .dial {
        --pct: 67;
        --dial-color: #E24B4A;
        width: 220px;
        height: 120px;
        border-top-left-radius: 220px;
        border-top-right-radius: 220px;
        overflow: hidden;
        position: relative;
        background:
            conic-gradient(from 180deg, var(--dial-color) calc(var(--pct) * 1.8deg), #f1efe8 0deg);
    }
    .dial::before {
        content: "";
        position: absolute;
        left: 22px;
        right: 22px;
        top: 22px;
        bottom: -98px;
        background: white;
        border-top-left-radius: 180px;
        border-top-right-radius: 180px;
    }
    .dial::after {
        content: "";
        position: absolute;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #141414;
        left: calc(50% + (var(--pct) - 50) * 1.35px - 8px);
        top: 12px;
        box-shadow: 0 0 0 4px rgba(255,255,255,.96);
    }
    .risk-num { font-size: 56px; font-weight: 600; color: #1f2d46; line-height: 1; }
    .risk-sub { font-size: 13px; color: #667085; margin-top: 6px; }
    .risk-pill { display: inline-block; margin-top: 10px; padding: 4px 14px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .pill-low { background: #e6f9f0; color: #2b8a50; }
    .pill-mod { background: #fff4e5; color: #b86908; }
    .pill-high { background: #ffe7e7; color: #d72e30; }
    .pill-calm { background: #e6f7ff; color: #0d7ea2; }
    .pill-action { background: #fff4e5; color: #b86908; }
    .pill-urgent { background: #ffe7e7; color: #d72e30; }
    .metric-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #667085; margin-bottom: 6px; }
    .metric-value { font-size: 30px; font-weight: 700; color: #1f2d46; }
    .metric-note { font-size: 12px; color: #667085; margin-top: 4px; }
    .framing-box { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: #fff; margin: 0.5rem 0 1rem; }
    .framing-summary { font-size: 14px; color: #1f2d46; line-height: 1.7; margin-bottom: 12px; }
    .framing-pills { display: flex; gap: 8px; flex-wrap: wrap; }
    .bar-row { margin: 12px 0; }
    .bar-head { display: flex; justify-content: space-between; font-size: 13px; color: #1f2d46; margin-bottom: 4px; }
    .bar-track { width: 100%; height: 8px; background: #eef2f7; border-radius: 999px; overflow: hidden; }
    .bar-risk { height: 100%; background: #E24B4A; }
    .bar-protect { height: 100%; background: #639922; }
    .action-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; background: #fff; }
    .action-badge { display: inline-block; margin-bottom: 6px; padding: 3px 8px; border-radius: 6px; font-size: 10px; font-weight: 700; }
    .badge-impact { background: #ffe7e7; color: #d72e30; }
    .badge-fast { background: #e6f9f0; color: #2b8a50; }
    .badge-doctor { background: #fff4e5; color: #b86908; }
    .badge-mod { background: #e6f7ff; color: #0d7ea2; }
    .action-title { font-size: 14px; font-weight: 600; color: #1f2d46; margin-bottom: 4px; }
    .action-desc { font-size: 13px; color: #667085; }
    .action-delta { font-size: 12px; color: #2b8a50; font-weight: 600; margin-top: 6px; }
    .prov-item { display: flex; align-items: center; gap: 8px; margin: 8px 0; font-size: 13px; color: #1f2d46; }
    .prov-tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }
    .prov-user { background: #e6f9f0; color: #2b8a50; }
    .prov-est { background: #e6f7ff; color: #0d7ea2; }
    </style>
    """, unsafe_allow_html=True)

    if st.button("← Back to calculator"):
        st.session_state["page"] = "input"
        st.rerun()

    if "results_data" in st.session_state:
        data = st.session_state["results_data"]
        risk_pct = data["risk_pct"]
        risk_label_ui = data["risk_label"]
        risk_badge_class = data["risk_badge_class"]
        risk_bar_class = "pill-low" if risk_badge_class == "risk-low" else "pill-mod" if risk_badge_class == "risk-mod" else "pill-high"
        dial_color = "#639922" if risk_badge_class == "risk-low" else "#EF9F27" if risk_badge_class == "risk-mod" else "#E24B4A"

        st.markdown('<div class="results-wrap">', unsafe_allow_html=True)

        tab_results, tab_breakdown, tab_whatif, tab_next, tab_population, tab_progress, tab_export = st.tabs(
            ["Results", "Risk breakdown", "What if?", "Next steps", "Population", "Progress", "Export PDF"]
        )

        with tab_results:
            st.markdown(
                f'<div class="risk-hero"><div class="dial-wrap"><div class="dial" style="--pct:{risk_pct}; --dial-color:{dial_color};"></div></div><div class="risk-num">{risk_pct}%</div><div class="risk-sub">estimated diabetes risk probability</div><span class="risk-pill {risk_bar_class}">{risk_label_ui}</span></div>',
                unsafe_allow_html=True,
            )

            pills_html = "".join(
                f'<span class="risk-pill {pill["class"]}">{pill["text"]}</span>' for pill in data.get("urgency_pills", [])
            )
            st.markdown(
                f'<div class="framing-box"><div class="framing-summary">{data["framing"]}</div><div class="framing-pills">{pills_html}</div></div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Fasting glucose</div><div class="metric-value">{data["glucose_display"]} <span style="font-size:16px;font-weight:500">mg/dL</span></div><div class="metric-note">{"estimated" if data["glucose_estimated"] else "user supplied"}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Insulin resistance</div><div class="metric-value">{data["ir_tier"]}</div><div class="metric-note">HOMA-IR ~{data["homa_range"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">FINDRISC score</div><div class="metric-value">{data["findrisc"]} <span style="font-size:16px;font-weight:500">/ 26</span></div><div class="metric-note">{data["findrisc_label"]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f'Estimate confidence: {data["confidence_label"]}')
            st.progress(data["confidence_pct"] / 100)

            st.markdown("#### Ranked action plan")
            for item in data.get("actions", []):
                st.markdown(
                    f'<div class="action-card"><span class="action-badge {item["badge_class"]}">{item["badge"]}</span><div class="action-title">{item["title"]}</div><div class="action-desc">{item["desc"]}</div><div class="action-delta">{item["delta"]}</div></div>',
                    unsafe_allow_html=True,
                )

        with tab_breakdown:
            st.markdown(data["breakdown_headline"])

            st.markdown("##### Risk drivers")
            for factor in data.get("risk_factor_bars", []):
                st.markdown(
                    f'<div class="bar-row"><div class="bar-head"><span>{factor["name"]}</span><span>{factor["pct"]}%</span></div><div class="bar-track"><div class="bar-risk" style="width:{factor["pct"]}%"></div></div><div style="font-size:12px;color:#667085">{factor["note"]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("##### Protective factors")
            for factor in data.get("protect_factor_bars", []):
                st.markdown(
                    f'<div class="bar-row"><div class="bar-head"><span>{factor["name"]}</span><span>{factor["pct"]}%</span></div><div class="bar-track"><div class="bar-protect" style="width:{factor["pct"]}%"></div></div><div style="font-size:12px;color:#667085">{factor["note"]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("##### Value provenance")
            for p in data.get("provenance", []):
                source = p.get("src", "estimated") if isinstance(p, dict) else "estimated"
                label = p.get("label", str(p)) if isinstance(p, dict) else str(p)
                tag_class = "prov-user" if source == "user" else "prov-est"
                tag_text = "user" if source == "user" else "estimated"
                st.markdown(
                    f'<div class="prov-item"><span class="prov-tag {tag_class}">{tag_text}</span><span>{label}</span></div>',
                    unsafe_allow_html=True,
                )

        with tab_whatif:
            baseline = data.get("baseline", {})
            sim_waist = st.slider("Waist circumference", 65, 130, int(round(baseline.get("waist", data.get("waist", 96)))), key="sim_waist")
            sim_bmi = st.slider("BMI", 16.0, 45.0, float(baseline.get("bmi", data.get("bmi", 27.5))), 0.1, key="sim_bmi")
            sim_bp = st.slider("Blood pressure (diastolic)", 50, 120, int(round(baseline.get("bp", data.get("bp", 88)))), key="sim_bp")
            sim_gluc = st.slider("Glucose", 70, 250, int(round(baseline.get("gluc", data.get("glucose_raw", 200)))), key="sim_gluc")

            delta = 0
            delta += (baseline.get("waist", sim_waist) - sim_waist) * 0.25
            delta += (baseline.get("bmi", sim_bmi) - sim_bmi) * 1.0
            delta += (baseline.get("bp", sim_bp) - sim_bp) * 0.2
            delta += (baseline.get("gluc", sim_gluc) - sim_gluc) * 0.15
            sim_risk = int(max(1, min(99, round(risk_pct - delta))))

            st.metric(
                "Simulated risk",
                f"{sim_risk}%",
                f"{sim_risk - risk_pct:+d}% vs baseline",
                delta_color="inverse",
            )

        with tab_next:
            st.markdown("#### This week")
            for item in data.get("this_week", []):
                st.markdown(f"- {item}")

            st.markdown("#### What to ask your doctor")
            for item in data.get("ask_doctor", []):
                st.markdown(f"- {item}")

            st.markdown("#### Lab tests")
            for item in data.get("lab_tests", []):
                st.markdown(f"- {item}")

        # ── Population Comparison Tab ──
        with tab_population:
            st.markdown("#### How you compare to your age group")
            pctl = compute_percentiles(
                age=data.get("baseline", {}).get("age", 35),
                bmi=data.get("bmi", 25),
                bp_dia=data.get("bp", 80),
                waist_cm=data.get("waist"),
            )
            st.caption(f"Based on {pctl['n_in_group']:,} NHANES participants aged {pctl['age_group']}")

            def _pctl_bar(label, info):
                if info is None:
                    return
                p = info["percentile"]
                color = "#639922" if p < 50 else "#EF9F27" if p < 75 else "#E24B4A"
                val_str = f"{info['value']}"
                med_str = f"Median: {info['median']}" if "median" in info else ""
                st.markdown(
                    f'<div class="bar-row"><div class="bar-head"><span>{label}</span>'
                    f'<span style="color:{color};font-weight:600">{p}th percentile</span></div>'
                    f'<div class="bar-track"><div style="height:100%;background:{color};width:{p}%"></div></div>'
                    f'<div style="font-size:12px;color:#667085">Your value: {val_str} · {med_str}</div></div>',
                    unsafe_allow_html=True,
                )

            _pctl_bar("BMI", pctl.get("bmi"))
            _pctl_bar("Blood Pressure (diastolic)", pctl.get("bp"))
            _pctl_bar("Waist Circumference", pctl.get("waist"))

            st.markdown("---")
            st.info(
                "💡 **Higher percentile** means your value is above more people in your age group. "
                "For BMI, blood pressure, and waist circumference, **lower** percentiles are generally healthier."
            )

        # ── Progress Tracking Tab ──
        with tab_progress:
            st.markdown("#### Your risk trend over time")
            history = _load_history()

            if len(history) < 2:
                st.info("Your risk history will appear here after you run at least 2 assessments. Come back after your next check-in!")
            else:
                hist_df = pd.DataFrame(history)
                hist_df["date"] = pd.to_datetime(hist_df["date"])

                st.line_chart(hist_df.set_index("date")[["risk_pct"]], y_label="Risk %", x_label="Date")

                st.markdown("##### Assessment history")
                for entry in reversed(history[-10:]):
                    pct = entry["risk_pct"]
                    d = entry["date"][:10]
                    bmi_h = entry.get("bmi", "?")
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:8px;padding:10px 14px">'
                        f'<span style="font-weight:600;font-size:18px;color:#1f2d46">{pct}%</span>'
                        f'<span style="font-size:12px;color:#667085;margin-left:12px">{d}</span>'
                        f'<span style="font-size:12px;color:#667085;margin-left:12px">BMI {bmi_h}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if len(history) >= 2:
                first = history[0]["risk_pct"]
                last = history[-1]["risk_pct"]
                delta = last - first
                if delta < 0:
                    st.success(f"📉 Your risk has decreased by {abs(delta)} points since your first assessment!")
                elif delta > 0:
                    st.warning(f"📈 Your risk has increased by {delta} points. Consider reviewing the action plan.")
                else:
                    st.info("Your risk has remained stable.")

        # ── Export PDF Tab ──
        with tab_export:
            st.markdown("#### Export your results")
            st.markdown("Generate a one-page PDF summary to bring to your next doctor visit.")

            pdf_bytes = generate_report_pdf(data)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"glycocast_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption("The PDF includes your risk score, key metrics, top risk drivers, action plan, and questions for your doctor.")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.session_state["page"] = "input"
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS (upgraded V2 styling)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .model-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1f77b4;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .new-badge {
        display: inline-block;
        background: #d4edda;
        color: #28a745;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 8px;
        vertical-align: middle;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #856404;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    # Try V2 model first, fall back to V1
    try:
        with open("model_v2.pkl", "rb") as f:
            return pickle.load(f), "v2"
    except FileNotFoundError:
        with open("model.pkl", "rb") as f:
            return pickle.load(f), "v1"

model_data, model_version = load_model()

if model_version == "v2":
    model = model_data['model']
    feature_columns = model_data['feature_columns']
else:
    model = model_data
    feature_columns = None  # V1 uses the old feature order

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🩺 GlycoCast</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Personalized diabetes risk assessment powered by NHANES health data</p>', unsafe_allow_html=True)

if model_version == "v2":
    st.markdown(
        f'<p style="text-align: center;"><span class="model-badge">V2 Model • {model_data.get("n_samples", "5,000+"):,} participants • Multi-ethnic</span></p>',
        unsafe_allow_html=True
    )

# Info expander
with st.expander("ℹ️ What do these measurements mean?"):
    st.markdown("""
    **Glucose**: Blood sugar level from a fasting blood test (normal: 70-99 mg/dL)
    
    **Blood Pressure**: Diastolic pressure (normal: <80 mmHg) - this is the bottom number in a blood pressure reading like 120/80
    
    **Insulin**: Fasting insulin level (normal: 2-25 μU/mL)
    
    **BMI**: Body Mass Index = weight(kg) / height(m)²
    
    **Waist Circumference**: Measured around the narrowest part of your waist
    
    *If you don't have recent medical test results, the app will estimate values based on your health profile.*
    """)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Synced slider + number input
# ══════════════════════════════════════════════════════════════════════════════

def synced_slider_number(label, min_value, max_value, value, step=None, format=None, key=None):
    if key is None:
        key = label.replace(" ", "_").lower()
    state_key = f"{key}_value"
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"

    # Initialise all three keys before any widget is rendered so that
    # widgets are never given both a `value=` argument AND a pre-existing
    # session-state key (which causes the Streamlit warning).
    if state_key not in st.session_state:
        st.session_state[state_key] = value
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[state_key]
    if number_key not in st.session_state:
        st.session_state[number_key] = st.session_state[state_key]

    def _slider_changed():
        st.session_state[state_key] = st.session_state[slider_key]
        st.session_state[number_key] = st.session_state[state_key]

    def _number_changed():
        st.session_state[state_key] = st.session_state[number_key]
        st.session_state[slider_key] = st.session_state[state_key]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=format,
            key=slider_key,
            on_change=_slider_changed,
        )
    with col2:
        st.number_input(
            "",
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=format,
            key=number_key,
            on_change=_number_changed,
            label_visibility="collapsed"
        )
    return st.session_state[state_key]

# ══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════

# ── NEW: Demographics Section ──
st.markdown('<h2 class="section-header">👤 Demographics <span class="new-badge">NEW</span></h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox(
        "Sex",
        options=["Female", "Male"],
        index=0,
        help="Biological sex - affects baseline risk calculations"
    )

with col2:
    ethnicity = st.selectbox(
        "Ethnicity",
        options=["White", "Black", "Hispanic", "Asian", "Other"],
        index=0,
        help="Ethnicity affects diabetes risk patterns"
    )

# ── Basic Information ──
st.markdown('<h2 class="section-header">📊 Basic Information</h2>', unsafe_allow_html=True)

# Only show pregnancies for females
if sex == "Female":
    pregnancies = synced_slider_number("Pregnancies", 0, 20, 1, step=1, key="pregnancies")
else:
    pregnancies = 0

col1, col2 = st.columns(2)
with col1:
    if weight_unit == "lbs":
        weight = synced_slider_number("Weight (lbs)", 88, 440, 154, step=1, key="weight")
        weight_kg = round(lbs_to_kg(weight), 1)
    else:
        weight = synced_slider_number("Weight (kg)", 40, 200, 70, step=1, key="weight")
        weight_kg = float(weight)
with col2:
    if length_unit == "inches":
        height = synced_slider_number("Height (inches)", 48, 84, 66, step=1, key="height")
        height_m = round(inches_to_cm(height) / 100, 3)
    else:
        height = synced_slider_number("Height (cm)", 120, 215, 170, step=1, key="height")
        height_m = round(height / 100, 3)

bmi = round(weight_kg / (height_m ** 2), 1) if height_m > 0 else 25.0
st.markdown(f"**Calculated BMI: {bmi} kg/m²**")

age = synced_slider_number("Age (years)", 18, 100, 35, step=1, key="age")

col1, col2 = st.columns(2)
with col1:
    blood_pressure = synced_slider_number("Blood Pressure - Diastolic (mmHg)", 50, 120, 80, step=1, key="blood_pressure")
with col2:
    if length_unit == "inches":
        waist_raw = synced_slider_number("Waist Circumference (inches)", 20, 60, 33, step=1, key="waist")
        waist_circumference = round(inches_to_cm(waist_raw), 1)
    else:
        waist_circumference = synced_slider_number("Waist Circumference (cm)", 50, 150, 85, step=1, key="waist")

# ── NEW: Medical History ──
st.markdown('<h2 class="section-header">🏥 Medical History <span class="new-badge">NEW</span></h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    high_bp = st.checkbox("High blood pressure", help="Ever been told you have high BP?")
    stroke_history = st.checkbox("History of stroke")
    
with col2:
    high_chol = st.checkbox("High cholesterol", help="Ever been told you have high cholesterol?")
    heart_disease = st.checkbox("Heart disease/attack")
    
with col3:
    prediabetes_diagnosed = st.checkbox("Prediabetes diagnosis", help="Ever been told you have prediabetes?")

# ── NEW: Lifestyle ──
st.markdown('<h2 class="section-header">🏃 Lifestyle <span class="new-badge">NEW</span></h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    phys_active = st.checkbox("Physically active (past 30 days)", value=True, help="Any exercise in the past month?")
    fruits_daily = st.checkbox("Eat fruit daily", value=True)
    veggies_daily = st.checkbox("Eat vegetables daily", value=True)
    
with col2:
    smoker = st.checkbox("Smoker (100+ cigarettes lifetime)", help="Have you smoked 100+ cigarettes ever?")
    heavy_alcohol = st.checkbox("Heavy alcohol consumption")
    sleep_trouble = st.checkbox("Sleep trouble", help="Difficulty falling or staying asleep?")

# Map activity/diet to your existing format
if phys_active:
    physical_activity = "Active"
else:
    physical_activity = "Sedentary"

if fruits_daily and veggies_daily:
    diet_quality = "Good"
elif fruits_daily or veggies_daily:
    diet_quality = "Average"
else:
    diet_quality = "Poor"

# ── Additional Health Factors (your original) ──
st.markdown('<h2 class="section-header">👨‍👩‍👧 Family History</h2>', unsafe_allow_html=True)

family_history = st.selectbox(
    "Family History of Diabetes",
    ["None", "One parent or sibling", "Both parents or early onset"],
    index=0,
    help="Both parents or early onset (<40 years) increases risk significantly"
)

# ── Optional Lab Values ──
st.markdown('<h2 class="section-header">🧪 Optional: Lab Values</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    know_glucose = st.checkbox("I know my glucose value")
    glucose = None
    if know_glucose:
        if glucose_unit == "mmol/L":
            glucose_raw = synced_slider_number("Glucose (mmol/L)", 2.8, 16.7, 5.6, step=0.1, format="%.1f", key="glucose")
            glucose = round(mmol_to_mgdl(glucose_raw), 1)
        else:
            glucose = synced_slider_number("Glucose (mg/dL)", 50, 300, 100, step=1, key="glucose")

with col2:
    know_insulin = st.checkbox("I know my insulin value")
    insulin = None
    if know_insulin:
        insulin = synced_slider_number("Insulin (µU/mL)", 0, 100, 10, step=1, key="insulin")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

if st.button("🔮 Predict My Risk", type="primary", use_container_width=True):
    
    # Convert display values to internal format
    family_history_map = {
        "None": "none",
        "One parent or sibling": "one parent or sibling",
        "Both parents or early onset": "both parents or early onset"
    }
    
    # Create PatientInputs object (your existing proxy system)
    patient = PatientInputs(
        pregnancies=pregnancies,
        bmi=bmi,
        age=age,
        blood_pressure=blood_pressure,
        waist_circumference=waist_circumference,
        physical_activity=physical_activity,
        diet_quality=diet_quality,
        family_history=family_history_map[family_history],
        prediabetes_diagnosed=prediabetes_diagnosed,
        glucose=glucose,
        insulin=insulin
    )

    # Build feature vector using your proxy system
    features = build_feature_vector(patient)

    if model_version == "v2":
        # V2 model with new features
        sex_code = 1 if sex == "Male" else 0
        ethnicity_map = {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other": 4}
        ethnicity_code = ethnicity_map[ethnicity]
        
        # General health from diet/activity
        if diet_quality == "Good" and physical_activity == "Active":
            gen_health = 2  # Very good
        elif diet_quality == "Poor" or physical_activity == "Sedentary":
            gen_health = 4  # Fair
        else:
            gen_health = 3  # Good
        
        input_data = pd.DataFrame([[
            pregnancies,
            features['bmi'],
            features['blood_pressure'],
            features['age'],
            sex_code,
            ethnicity_code,
            int(high_bp),
            int(high_chol),
            int(phys_active),
            int(smoker),
            gen_health,
            int(sleep_trouble)
        ]], columns=feature_columns)
        
        probability = model.predict_proba(input_data)[0][1]
    else:
        # V1 model (original Pima features)
        input_data = pd.DataFrame([[
            features['pregnancies'],
            features['glucose'],
            features['blood_pressure'],
            features['skin_thickness'],
            features['insulin'],
            features['bmi'],
            features['diabetes_pedigree'],
            features['age']
        ]], columns=[
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ])
        
        probability = model.predict_proba(input_data)[0][1]

    risk_pct = min(max(int(round(probability * 100)), 0), 100)

    # Generate results content using your existing system
    content = generate_results_content(patient, features, risk_pct)
    content["baseline"] = {
        "waist": float(patient.waist_circumference),
        "bmi": float(patient.bmi),
        "bp": float(patient.blood_pressure),
        "gluc": float(features['glucose']),
        "age": int(patient.age),
    }
    
    # Add new demographic info to results
    content["sex"] = sex
    content["ethnicity"] = ethnicity

    # ── Save to progress history ──
    history = _load_history()
    history.append({
        "date": datetime.now().isoformat(),
        "risk_pct": risk_pct,
        "bmi": round(patient.bmi, 1),
        "bp": round(patient.blood_pressure, 1),
        "waist": round(patient.waist_circumference, 1),
        "glucose": round(features['glucose'], 1),
        "age": patient.age,
    })
    _save_history(history)

    st.session_state["results_data"] = content
    st.session_state["page"] = "results"
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="disclaimer">
<strong>⚠️ Medical Disclaimer</strong><br><br>
This tool provides an <strong>estimate only</strong> based on population health data. 
It is <strong>not a medical diagnosis</strong>. Please consult a healthcare provider 
for proper diabetes screening, especially if your estimated risk is elevated.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
if model_version == "v2":
    st.markdown(
        f"<p style='text-align: center; color: #888; font-size: 0.85rem;'>"
        f"GlycoCast v2.0 • NHANES data • {model_data.get('n_samples', 'N/A'):,} participants • "
        f"Accuracy: {model_data.get('accuracy', 0):.1%}"
        "</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.85rem;'>"
        "GlycoCast v1.0 • Pima dataset"
        "</p>",
        unsafe_allow_html=True
    )
