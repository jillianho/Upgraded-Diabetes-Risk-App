from dataclasses import dataclass
from typing import Optional

# Proxy functions in this file are heuristic approximations intended for
# educational risk exploration only. They are not diagnostic substitutions for
# fasting lab tests. FINDRISC-inspired rules are adapted from public screening
# guidance and simplified for this app's user input surface.


@dataclass
class PatientInputs:
    pregnancies: int
    bmi: float
    age: int
    blood_pressure: float       # diastolic mmHg
    waist_circumference: float  # cm
    physical_activity: str      # "Active", "Moderate", "Sedentary"
    diet_quality: str           # "Good", "Average", "Poor"
    family_history: str         # "none", "one parent or sibling", "both parents or early onset"
    prediabetes_diagnosed: bool
    glucose: Optional[float] = None
    insulin: Optional[float] = None


def build_feature_vector(patient: PatientInputs) -> dict:
    glucose_val = patient.glucose if patient.glucose is not None else _estimate_glucose(patient)
    insulin_val = patient.insulin if patient.insulin is not None else _estimate_insulin(patient)

    features = {
        'pregnancies':        patient.pregnancies,
        'glucose':            round(glucose_val, 1),
        'blood_pressure':     patient.blood_pressure,
        'skin_thickness':     _estimate_skin_thickness(patient),
        'insulin':            round(insulin_val, 1),
        'bmi':                patient.bmi,
        'diabetes_pedigree':  _estimate_diabetes_pedigree(patient),
        'age':                patient.age,
        '_glucose_estimated': patient.glucose is None,
        '_insulin_estimated': patient.insulin is None,
    }
    return features


# ---------------------------------------------------------------------------
# Glucose proxy — mirrors actual Pima dataset distribution (44–199 mg/dL)
# ---------------------------------------------------------------------------
def _estimate_glucose(patient: PatientInputs) -> float:
    score = _findrisc_score(patient)
    if score < 7:
        base = 88.0
    elif score < 12:
        base = 98.0
    elif score < 15:
        base = 110.0
    elif score < 20:
        base = 128.0
    else:
        base = 152.0

    if patient.prediabetes_diagnosed:
        base += 22.0

    if patient.bmi >= 40:
        base += 18.0
    elif patient.bmi >= 35:
        base += 12.0
    elif patient.bmi >= 30:
        base += 6.0

    if patient.waist_circumference >= 115:
        base += 12.0
    elif patient.waist_circumference >= 102:
        base += 7.0
    elif patient.waist_circumference >= 88:
        base += 3.0

    diet_adj = {"Poor": 8.0, "Average": 2.0, "Good": -3.0}
    base += diet_adj.get(patient.diet_quality, 0.0)

    act_adj = {"Sedentary": 6.0, "Moderate": 1.0, "Active": -4.0}
    base += act_adj.get(patient.physical_activity, 0.0)

    if patient.age >= 60:
        base += 8.0
    elif patient.age >= 45:
        base += 4.0

    return min(max(base, 60.0), 199.0)


# ---------------------------------------------------------------------------
# Insulin proxy — high-risk cases go up to 200+ µU/mL as in Pima dataset
# ---------------------------------------------------------------------------
def _estimate_insulin(patient: PatientInputs) -> float:
    base = 12.0

    if patient.bmi >= 40:
        base += 80.0
    elif patient.bmi >= 35:
        base += 55.0
    elif patient.bmi >= 30:
        base += 35.0
    elif patient.bmi >= 25:
        base += 15.0

    if patient.waist_circumference >= 115:
        base += 40.0
    elif patient.waist_circumference >= 102:
        base += 25.0
    elif patient.waist_circumference >= 88:
        base += 12.0

    if patient.prediabetes_diagnosed:
        base += 30.0

    act_adj = {"Sedentary": 20.0, "Moderate": 8.0, "Active": 0.0}
    base += act_adj.get(patient.physical_activity, 0.0)

    diet_adj = {"Poor": 15.0, "Average": 5.0, "Good": 0.0}
    base += diet_adj.get(patient.diet_quality, 0.0)

    if patient.family_history == "both parents or early onset":
        base += 20.0
    elif patient.family_history == "one parent or sibling":
        base += 10.0

    if patient.age >= 50:
        base += 10.0
    elif patient.age >= 40:
        base += 5.0

    return min(max(base, 5.0), 250.0)


# ---------------------------------------------------------------------------
# Skin thickness proxy
# ---------------------------------------------------------------------------
def _estimate_skin_thickness(patient: PatientInputs) -> float:
    raw = 0.31 * patient.waist_circumference + 0.10 * patient.bmi - 10.4
    act_adj = {"Active": -5.0, "Moderate": -2.0, "Sedentary": 0.0}
    raw += act_adj.get(patient.physical_activity, 0.0)
    return round(min(max(raw, 7.0), 60.0), 1)


# ---------------------------------------------------------------------------
# Diabetes pedigree function proxy
# ---------------------------------------------------------------------------
def _estimate_diabetes_pedigree(patient: PatientInputs) -> float:
    base = {"none": 0.18, "one parent or sibling": 0.52, "both parents or early onset": 1.05}.get(patient.family_history, 0.18)
    if patient.age >= 45:
        base *= 1.2
    elif patient.age >= 35:
        base *= 1.1
    return round(min(base, 2.5), 3)


# ---------------------------------------------------------------------------
# FINDRISC score (internal helper)
# ---------------------------------------------------------------------------
def _findrisc_score(patient: PatientInputs) -> int:
    score = 0
    if patient.age >= 65:       score += 4
    elif patient.age >= 55:     score += 3
    elif patient.age >= 45:     score += 2

    if patient.bmi >= 30:       score += 3
    elif patient.bmi >= 25:     score += 1

    if patient.waist_circumference >= 102:  score += 4
    elif patient.waist_circumference >= 88: score += 3

    if patient.physical_activity == "Sedentary": score += 2
    if patient.diet_quality == "Poor":           score += 1
    if patient.blood_pressure >= 90:             score += 2

    if patient.family_history == "both parents or early onset": score += 5
    elif patient.family_history == "one parent or sibling":     score += 3

    if patient.prediabetes_diagnosed: score += 5

    return score


# ---------------------------------------------------------------------------
# Dynamic results content generator
# ---------------------------------------------------------------------------
def generate_results_content(patient: PatientInputs, features: dict, risk_pct: int) -> dict:
    glucose  = features['glucose']
    insulin  = features['insulin']
    bmi      = patient.bmi
    waist    = patient.waist_circumference
    age      = patient.age
    bp       = patient.blood_pressure
    act      = patient.physical_activity
    diet     = patient.diet_quality
    fh       = patient.family_history
    prediab  = patient.prediabetes_diagnosed
    g_est    = features['_glucose_estimated']
    i_est    = features['_insulin_estimated']
    findrisc = _findrisc_score(patient)

    # Risk tier
    if risk_pct < 30:
        risk_tier = "low"; risk_label = "Low risk"; risk_badge_class = "risk-low"
    elif risk_pct < 60:
        risk_tier = "moderate"; risk_label = "Moderate risk"; risk_badge_class = "risk-mod"
    else:
        risk_tier = "high"; risk_label = "High risk"; risk_badge_class = "risk-high"

    # IR tier
    if insulin < 15:
        ir_tier = "Normal"; homa_range = "< 1.9"
    elif insulin < 30:
        ir_tier = "Borderline"; homa_range = "1.9–2.9"
    else:
        ir_tier = "Resistant"; homa_range = "2.9–5.0"

    # FINDRISC label
    if findrisc < 7:    findrisc_label = "low risk"
    elif findrisc < 12: findrisc_label = "slightly elevated"
    elif findrisc < 15: findrisc_label = "moderate"
    elif findrisc < 20: findrisc_label = "high"
    else:               findrisc_label = "very high"

    # Drivers
    drivers = []
    if glucose > 140:       drivers.append("critically elevated glucose")
    elif glucose > 110:     drivers.append("elevated glucose")
    if waist >= 102:        drivers.append("high central adiposity")
    elif waist >= 88:       drivers.append("central adiposity")
    if bmi >= 30:           drivers.append("obesity")
    elif bmi >= 25:         drivers.append("overweight BMI")
    if prediab:             drivers.append("prior prediabetes diagnosis")
    if fh != "none":        drivers.append("family history")
    if bp >= 90:            drivers.append("elevated blood pressure")
    if act == "Sedentary":  drivers.append("sedentary lifestyle")

    modifiable = [d for d in drivers if d not in ("family history", "prior prediabetes diagnosis")]

    # Emotional framing
    if risk_tier == "low":
        framing = (
            f"Your profile looks relatively healthy. "
            f"{'Your main protective factors are active lifestyle and good diet. ' if act == 'Active' and diet == 'Good' else ''}"
            f"Staying consistent with your current lifestyle is the most important thing you can do."
        )
        urgency_pills = [
            {"text": "Low urgency", "class": "pill-calm"},
            {"text": "Maintain current habits", "class": "pill-calm"},
        ]
    elif risk_tier == "moderate":
        main_driver = drivers[0] if drivers else "multiple factors"
        framing = (
            f"Your risk is primarily driven by {main_driver}. "
            f"{'Several of your risk factors are modifiable — your outlook can change meaningfully with targeted action.' if modifiable else 'Some risk factors are genetic, but lifestyle changes can still reduce your overall risk.'}"
        )
        urgency_pills = [
            {"text": "Not a diagnosis", "class": "pill-calm"},
            {"text": f"Main lever: {modifiable[0] if modifiable else 'lifestyle'}", "class": "pill-action"},
        ]
        if glucose > 110 or prediab:
            urgency_pills.append({"text": "Worth discussing with a clinician", "class": "pill-action"})
    else:
        main_driver = drivers[0] if drivers else "multiple compounding factors"
        framing = (
            f"Your risk is high, driven primarily by {main_driver}. "
            f"{'This does not mean you have diabetes — it means your risk profile warrants action. ' if not prediab else 'Given your prediabetes history, this result needs clinical follow-up. '}"
            f"{'You have meaningful modifiable levers: ' + ', '.join(modifiable[:2]) + '.' if modifiable else 'Focus on the actions below.'}"
        )
        urgency_pills = [
            {"text": "High risk — not a diagnosis", "class": "pill-calm"},
            {"text": f"Priority: {modifiable[0] if modifiable else 'clinical review'}", "class": "pill-action"},
        ]
        if glucose > 140 or prediab:
            urgency_pills.append({"text": "Discuss with a clinician soon", "class": "pill-urgent"})

    # Confidence
    estimated_count = sum([g_est, i_est, True])
    if estimated_count <= 1:
        confidence_pct = 82; confidence_label = "High — most values user-supplied"
    elif estimated_count == 2:
        confidence_pct = 65; confidence_label = "Medium — some values estimated"
    else:
        confidence_pct = 48; confidence_label = "Lower — several values estimated from proxies"

    # Ranked actions
    actions = []
    if glucose > 126 or prediab:
        actions.append({
            "badge": "See a clinician", "badge_class": "badge-doctor",
            "title": "Get a fasting plasma glucose or HbA1c test",
            "desc": f"Your {'reported' if not g_est else 'estimated'} glucose of {glucose:.0f} mg/dL is {'in the diabetic range' if glucose > 126 else 'elevated'}. This is the most important thing to confirm with a real lab test.",
            "delta": "Required for accurate diagnosis"
        })
    if waist >= 88:
        actions.append({
            "badge": "Highest impact", "badge_class": "badge-impact",
            "title": f"Reduce waist circumference ({waist:.0f} cm → target <88 cm)",
            "desc": "Central fat is the strongest modifiable driver of insulin resistance. Even a 5 cm reduction meaningfully shifts your risk.",
            "delta": f"Estimated reduction: ~{'6' if waist < 102 else '10'}%"
        })
    if act != "Active":
        actions.append({
            "badge": "Fastest win", "badge_class": "badge-fast",
            "title": "Daily 20-min walks after meals",
            "desc": "Post-meal walking reduces glucose spikes by 20–30%. Measurable change within 2 weeks.",
            "delta": "Estimated reduction: ~5–7%"
        })
    if act == "Sedentary":
        actions.append({
            "badge": "Good leverage", "badge_class": "badge-mod",
            "title": "Increase to moderate activity (30+ min most days)",
            "desc": "Moving from sedentary to moderate shifts 2 FINDRISC points and improves insulin sensitivity.",
            "delta": "Estimated reduction: ~9%"
        })
    if diet == "Poor":
        actions.append({
            "badge": "Good leverage", "badge_class": "badge-mod",
            "title": "Reduce refined carbs and ultra-processed foods",
            "desc": "Poor diet adds ~8 mg/dL to estimated glucose and 1 FINDRISC point.",
            "delta": "Estimated reduction: ~4%"
        })
    elif diet == "Average":
        actions.append({
            "badge": "Good leverage", "badge_class": "badge-mod",
            "title": "Improve diet to mostly whole foods",
            "desc": "Shifting from average to good diet reduces estimated glucose by ~5 mg/dL.",
            "delta": "Estimated reduction: ~3%"
        })
    if bp >= 85:
        actions.append({
            "badge": "Good leverage", "badge_class": "badge-mod",
            "title": f"Reduce blood pressure ({bp:.0f} mmHg diastolic)",
            "desc": "Diastolic ≥85 adds 2 FINDRISC points. Reducing sodium and increasing aerobic activity helps.",
            "delta": "Estimated reduction: ~4%"
        })
    if bmi >= 25:
        actions.append({
            "badge": "Long term", "badge_class": "badge-mod",
            "title": f"Reduce BMI ({bmi:.1f} → target <25)",
            "desc": f"BMI of {bmi:.1f} {'is obese' if bmi >= 30 else 'is overweight'}. Each unit reduction lowers insulin resistance.",
            "delta": "Estimated reduction: ~3–8%"
        })
    if not actions:
        actions.append({
            "badge": "Maintain", "badge_class": "badge-fast",
            "title": "Keep up your current healthy habits",
            "desc": "Your profile shows low risk. Regular check-ups every 2–3 years are sufficient.",
            "delta": "No immediate action required"
        })

    # Factor bars
    risk_factor_bars = []
    protect_factor_bars = []

    def add_factor(name, pct, note, is_risk):
        (risk_factor_bars if is_risk else protect_factor_bars).append(
            {"name": name, "pct": min(max(pct, 5), 95), "note": note}
        )

    if glucose > 140:    add_factor(f"Glucose ({glucose:.0f} mg/dL)", 90, "Critical — in diabetic range", True)
    elif glucose > 110:  add_factor(f"Glucose ({glucose:.0f} mg/dL)", 65, "Elevated above normal", True)
    elif glucose > 99:   add_factor(f"Glucose ({glucose:.0f} mg/dL)", 40, "Borderline elevated", True)
    else:                add_factor(f"Glucose ({glucose:.0f} mg/dL)", 55, "Within normal range", False)

    if waist >= 102:     add_factor(f"Waist ({waist:.0f} cm)", 75, "High — strong visceral fat signal", True)
    elif waist >= 88:    add_factor(f"Waist ({waist:.0f} cm)", 50, "Above threshold", True)
    else:                add_factor(f"Waist ({waist:.0f} cm)", 45, "Within healthy range", False)

    if bmi >= 35:        add_factor(f"BMI ({bmi:.1f})", 70, "Obese class II+", True)
    elif bmi >= 30:      add_factor(f"BMI ({bmi:.1f})", 55, "Obese", True)
    elif bmi >= 25:      add_factor(f"BMI ({bmi:.1f})", 30, "Overweight", True)
    else:                add_factor(f"BMI ({bmi:.1f})", 50, "Healthy BMI", False)

    if age >= 55:        add_factor(f"Age ({age})", 65, "Age 55+ significantly increases risk", True)
    elif age >= 45:      add_factor(f"Age ({age})", 45, "Risk rises from age 45", True)
    else:                add_factor(f"Age ({age})", 40, "Younger age is protective", False)

    if bp >= 90:         add_factor(f"Blood pressure ({bp:.0f} mmHg)", 50, "Elevated — adds 2 FINDRISC points", True)
    elif bp >= 85:       add_factor(f"Blood pressure ({bp:.0f} mmHg)", 35, "Borderline elevated", True)
    else:                add_factor(f"Blood pressure ({bp:.0f} mmHg)", 35, "Normal range", False)

    if act == "Active":      add_factor("Physical activity (active)", 60, "Regular activity reduces IR", False)
    elif act == "Moderate":  add_factor("Physical activity (moderate)", 30, "Some protection vs sedentary", False)
    else:                    add_factor("Physical activity (sedentary)", 55, "Adds 2 FINDRISC points", True)

    if diet == "Good":   add_factor("Diet quality (good)", 50, "Reduces glucose and IR", False)
    elif diet == "Poor": add_factor("Diet quality (poor)", 45, "Adds ~8 mg/dL to glucose estimate", True)

    if fh == "both parents or early onset":  add_factor("Family history (both parents)", 60, "Strongest genetic signal", True)
    elif fh == "one parent or sibling":      add_factor("Family history (one parent)", 35, "Moderate genetic contribution", True)
    else:                                    add_factor("No family history", 40, "Protective", False)

    if prediab:  add_factor("Prior prediabetes diagnosis", 70, "Clinical confirmation of elevated risk", True)
    else:        add_factor("No prediabetes diagnosis", 35, "No prior clinical flag", False)

    risk_factor_bars.sort(key=lambda x: x['pct'], reverse=True)
    protect_factor_bars.sort(key=lambda x: x['pct'], reverse=True)

    # Breakdown headline
    if risk_factor_bars:
        top = risk_factor_bars[0]['name'].split('(')[0].strip()
        if len(risk_factor_bars) > 1:
            second = risk_factor_bars[1]['name'].split('(')[0].strip().lower()
            breakdown_headline = f"{top} is your dominant risk driver, followed by {second}. {'Improving activity and diet are your highest-leverage actions.' if act != 'Active' else 'Your activity level is your strongest protective factor.'}"
        else:
            breakdown_headline = f"{top} is your primary risk driver. Your other factors are relatively well-controlled."
    else:
        breakdown_headline = "Your risk factors are well-controlled. Maintaining your current lifestyle is the priority."

    # Next steps
    this_week = []
    ask_doctor = []
    lab_tests = []

    if glucose > 126 or prediab:
        this_week.append("Book a fasting plasma glucose or HbA1c test — urgent given your glucose level")
    if waist >= 88:
        this_week.append(f"Record your waist circumference weekly — current {waist:.0f} cm, target <88 cm")
    if act != "Active":
        this_week.append("Start 20-minute post-meal walks — fastest impact on glucose")
    if diet == "Poor":
        this_week.append("Replace one ultra-processed meal per day with whole foods")

    if glucose > 110 or prediab:
        ask_doctor.append(f"\"Should I get an HbA1c test given my glucose of {glucose:.0f} mg/dL?\"")
    if bp >= 85:
        ask_doctor.append(f"\"Is my blood pressure ({bp:.0f} mmHg diastolic) worth monitoring?\"")
    if fh != "none":
        ask_doctor.append("\"Given my family history, should I be screened more frequently?\"")
    ask_doctor.append("\"Am I a candidate for a structured diabetes prevention programme?\"")

    if i_est:
        lab_tests.append("Fasting insulin (would replace our estimated IR tier with real data)")
    if g_est:
        lab_tests.append("Fasting plasma glucose (most direct confirmation)")
    lab_tests.append("HbA1c (3-month average — more reliable than a single glucose reading)")
    lab_tests.append("Waist-to-height ratio (stronger predictor than BMI alone)")

    # Provenance
    provenance = [
        {"src": "user" if not g_est else "estimated",
         "label": f"Glucose: {glucose:.0f} mg/dL {'(entered directly)' if not g_est else '— estimated from FINDRISC + lifestyle'}"},
        {"src": "user" if not i_est else "estimated",
         "label": f"Insulin: {features['insulin']:.0f} µU/mL {'(entered directly)' if not i_est else '— estimated from BMI, waist, activity, family history'}"},
        {"src": "estimated", "label": f"Skin thickness: {features['skin_thickness']:.1f} mm — from waist + BMI regression"},
        {"src": "user", "label": f"Blood pressure: {bp:.0f} mmHg (entered directly)"},
        {"src": "user", "label": f"BMI: {bmi:.1f} (entered directly)"},
        {"src": "user", "label": f"Pregnancies: {patient.pregnancies} (entered directly)"},
        {"src": "estimated", "label": f"DiabetesPedigreeFunction: {features['diabetes_pedigree']:.2f} — from family history ({fh})"},
    ]

    return {
        "risk_pct":             risk_pct,
        "risk_tier":            risk_tier,
        "risk_label":           risk_label,
        "risk_badge_class":     risk_badge_class,
        "glucose_display":      f"{glucose:.1f}",
        "glucose_estimated":    g_est,
        "ir_tier":              ir_tier,
        "homa_range":           homa_range,
        "findrisc":             findrisc,
        "findrisc_label":       findrisc_label,
        "framing":              framing,
        "urgency_pills":        urgency_pills,
        "confidence_pct":       confidence_pct,
        "confidence_label":     confidence_label,
        "actions":              actions,
        "risk_factor_bars":     risk_factor_bars,
        "protect_factor_bars":  protect_factor_bars,
        "breakdown_headline":   breakdown_headline,
        "this_week":            this_week,
        "ask_doctor":           ask_doctor,
        "lab_tests":            lab_tests,
        "provenance":           provenance,
        "waist":                waist,
        "bmi":                  bmi,
        "bp":                   bp,
        "glucose_raw":          glucose,
        "act":                  act,
    }
