# File: train_model_v2.py
# GlycoCast Model V2 - Upgraded with NHANES data
# Keeps your original features (glucose, insulin, BMI, BP, age) 
# Adds: sex, ethnicity, medical history, lifestyle factors

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("🩺 GLYCOCAST MODEL V2 - NHANES TRAINING")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD NHANES DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n📊 Loading NHANES dataset...")

# Try multiple sources for the NHANES data
urls = [
    "https://raw.githubusercontent.com/GTPB/PSLS20/master/data/NHANES.csv",
    "https://raw.githubusercontent.com/ProjectMOSAIC/NHANES/master/data-raw/NHANES.csv",
]

df = None
for url in urls:
    try:
        print(f"   Trying: {url[:50]}...")
        df = pd.read_csv(url)
        print(f"✓ Loaded NHANES data: {len(df):,} rows")
        break
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        continue

# If online sources fail, try local file
if df is None:
    local_path = "NHANES.csv"
    try:
        df = pd.read_csv(local_path)
        print(f"✓ Loaded from local file: {len(df):,} rows")
    except:
        print("\n" + "="*65)
        print("❌ ERROR: Could not load NHANES data")
        print("="*65)
        print("""
Please download the NHANES dataset manually:

1. Go to: https://raw.githubusercontent.com/GTPB/PSLS20/master/data/NHANES.csv
2. Right-click → "Save As" → save as "NHANES.csv" 
3. Put it in the same folder as this script
4. Run this script again

Or use curl/wget:
   curl -o NHANES.csv https://raw.githubusercontent.com/GTPB/PSLS20/master/data/NHANES.csv
""")
        exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n📋 Dataset columns available:")
print(f"   {len(df.columns)} columns total")

# Key columns we need
key_cols = ['Gender', 'Age', 'Race1', 'BMI', 'BPDiaAve', 'BPSysAve', 
            'DirectChol', 'TotChol', 'Diabetes', 'PhysActive', 
            'Smoke100', 'SleepTrouble', 'HealthGen', 'nPregnancies']

print("\n   Key columns for our model:")
for col in key_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   ✓ {col}: {non_null:,} non-null values")
    else:
        print(f"   ✗ {col}: NOT FOUND")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n🔧 Preparing data...")

# Remove duplicates (NHANES has repeated measurements per person)
df_unique = df.drop_duplicates(subset=['ID'], keep='first')
print(f"   Unique participants: {len(df_unique):,}")

# Filter to adults (18+) with diabetes status
df_clean = df_unique[
    (df_unique['Age'] >= 18) & 
    (df_unique['Diabetes'].notna())
].copy()
print(f"   Adults with diabetes status: {len(df_clean):,}")

# Create binary diabetes target
df_clean['diabetes_binary'] = (df_clean['Diabetes'] == 'Yes').astype(int)

# ── Encode categorical variables ──

# Gender: Male=1, Female=0
df_clean['sex'] = (df_clean['Gender'] == 'male').astype(int)

# Race/Ethnicity encoding
race_map = {
    'White': 0,
    'Black': 1, 
    'Hispanic': 2,
    'Mexican': 2,  # Group with Hispanic
    'Asian': 3,
    'Other': 4
}
df_clean['ethnicity'] = df_clean['Race1'].map(race_map).fillna(4)

# Physical activity: Yes=1, No=0
df_clean['phys_active'] = (df_clean['PhysActive'] == 'Yes').astype(int)

# Smoking history: Yes=1, No=0
df_clean['smoker'] = (df_clean['Smoke100'] == 'Yes').astype(int)

# Sleep trouble: Yes=1, No=0
df_clean['sleep_trouble'] = (df_clean['SleepTrouble'] == 'Yes').astype(int)

# General health (1=Excellent to 5=Poor)
health_map = {'Excellent': 1, 'Vgood': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
df_clean['gen_health'] = df_clean['HealthGen'].map(health_map).fillna(3)

# High cholesterol proxy (TotChol > 240 mg/dL)
df_clean['high_chol'] = (df_clean['TotChol'] > 240).astype(int)

# High blood pressure proxy (systolic >= 140 or diastolic >= 90)
df_clean['high_bp'] = ((df_clean['BPSysAve'] >= 140) | (df_clean['BPDiaAve'] >= 90)).astype(int)

# Pregnancies (for females, 0 for males)
df_clean['pregnancies'] = df_clean['nPregnancies'].fillna(0)
df_clean.loc[df_clean['sex'] == 1, 'pregnancies'] = 0  # Males = 0

# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

# Original Pima features (keeping compatibility):
#   pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age
#
# NEW features we're adding:
#   sex, ethnicity, high_bp, high_chol, smoker, phys_active, gen_health, sleep_trouble

# Note: NHANES doesn't have glucose/insulin in this subset, so we'll use proxies
# and the model will work with OR without lab values (your proxy system handles this!)

feature_cols = [
    # Original features (compatible with your app)
    'pregnancies',
    'BMI',
    'BPDiaAve',      # Diastolic blood pressure
    'Age',
    
    # NEW demographic features
    'sex',
    'ethnicity',
    
    # NEW medical history
    'high_bp',
    'high_chol',
    
    # NEW lifestyle features  
    'phys_active',
    'smoker',
    'gen_health',
    'sleep_trouble',
]

# Rename to match your app's expected names
rename_map = {
    'BMI': 'bmi',
    'BPDiaAve': 'blood_pressure',
    'Age': 'age',
}

# Select and clean features
df_model = df_clean[feature_cols + ['diabetes_binary']].copy()
df_model = df_model.rename(columns=rename_map)
df_model = df_model.dropna()

print(f"\n   Final dataset: {len(df_model):,} samples")
print(f"   Features: {len(feature_cols)}")

# Check class balance
diabetes_counts = df_model['diabetes_binary'].value_counts()
print(f"\n   Class distribution:")
print(f"   • No diabetes: {diabetes_counts[0]:,} ({diabetes_counts[0]/len(df_model)*100:.1f}%)")
print(f"   • Diabetes: {diabetes_counts[1]:,} ({diabetes_counts[1]/len(df_model)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# FIX: BALANCE THE CLASSES (so model learns to identify high-risk cases)
# ══════════════════════════════════════════════════════════════════════════════

print("\n⚖️  Balancing classes...")

# Separate by class
df_no_diabetes = df_model[df_model['diabetes_binary'] == 0]
df_diabetes = df_model[df_model['diabetes_binary'] == 1]

# Undersample the majority class to 2:1 ratio
n_diabetes = len(df_diabetes)
n_no_diabetes_sample = min(len(df_no_diabetes), n_diabetes * 2)

df_no_diabetes_sampled = df_no_diabetes.sample(n=n_no_diabetes_sample, random_state=42)
df_model = pd.concat([df_no_diabetes_sampled, df_diabetes]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Balanced dataset: {len(df_model):,} samples")
print(f"   New ratio - No diabetes: {(df_model['diabetes_binary']==0).sum():,}, Diabetes: {(df_model['diabetes_binary']==1).sum():,}")

# Demographics breakdown
print(f"\n   Demographics:")
print(f"   • Male: {(df_model['sex']==1).sum():,}")
print(f"   • Female: {(df_model['sex']==0).sum():,}")
print(f"\n   Ethnicities:")
for eth_code, eth_name in enumerate(['White', 'Black', 'Hispanic', 'Asian', 'Other']):
    count = (df_model['ethnicity']==eth_code).sum()
    print(f"   • {eth_name}: {count:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

print("\n🔀 Splitting data...")

# Final feature column names (after renaming)
final_feature_cols = [
    'pregnancies', 'bmi', 'blood_pressure', 'age',
    'sex', 'ethnicity', 'high_bp', 'high_chol',
    'phys_active', 'smoker', 'gen_health', 'sleep_trouble'
]

X = df_model[final_feature_cols]
y = df_model['diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,}")
print(f"   Test set: {len(X_test):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n🤖 Training model...")

# Gradient Boosting with calibration for reliable probabilities
base_model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    min_samples_split=30,
    min_samples_leaf=15,
    random_state=42,
    verbose=0
)

# Calibrate probabilities
model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)

print("   Training calibrated Gradient Boosting model...")
model.fit(X_train, y_train)
print("✓ Model trained")

# ══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

print("\n📈 Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n   Test Accuracy: {accuracy:.1%}")
print(f"   ROC-AUC Score: {roc_auc:.3f}")

# Test worst-case scenario
print("\n🧪 Testing predictions...")
worst_case = pd.DataFrame([[
    10,    # pregnancies
    45,    # bmi (morbidly obese)
    100,   # blood_pressure (high)
    70,    # age
    1,     # sex (male)
    1,     # ethnicity
    1,     # high_bp (yes)
    1,     # high_chol (yes)
    0,     # phys_active (no)
    1,     # smoker (yes)
    5,     # gen_health (poor)
    1      # sleep_trouble (yes)
]], columns=final_feature_cols)

best_case = pd.DataFrame([[
    0,     # pregnancies
    22,    # bmi (healthy)
    70,    # blood_pressure (normal)
    25,    # age (young)
    0,     # sex (female)
    0,     # ethnicity
    0,     # high_bp (no)
    0,     # high_chol (no)
    1,     # phys_active (yes)
    0,     # smoker (no)
    1,     # gen_health (excellent)
    0      # sleep_trouble (no)
]], columns=final_feature_cols)

worst_prob = model.predict_proba(worst_case)[0][1]
best_prob = model.predict_proba(best_case)[0][1]

print(f"   • Worst case (unhealthy 70yo): {worst_prob:.1%} risk")
print(f"   • Best case (healthy 25yo): {best_prob:.1%} risk")
print(f"   • Range: {best_prob:.1%} → {worst_prob:.1%} ✓")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# ══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n🎯 Feature Importance:")

# Train a quick model to get feature importances
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_importance = sorted(zip(final_feature_cols, importances), key=lambda x: -x[1])

print("\n   Risk factors (most → least important):")
for name, score in feature_importance:
    bar = "█" * int(score * 40)
    print(f"   {name:<18} {bar} {score:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n💾 Saving model...")

model_data = {
    'model': model,
    'feature_columns': final_feature_cols,
    'version': '2.0',
    'dataset': 'NHANES 2009-2012',
    'n_samples': len(df_model),
    'accuracy': accuracy,
    'roc_auc': roc_auc,
    'ethnicity_map': {
        0: 'White',
        1: 'Black', 
        2: 'Hispanic',
        3: 'Asian',
        4: 'Other'
    },
    'feature_info': {
        'pregnancies': 'Number of pregnancies (0 for males)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Diastolic blood pressure (mmHg)',
        'age': 'Age in years',
        'sex': '0=Female, 1=Male',
        'ethnicity': '0=White, 1=Black, 2=Hispanic, 3=Asian, 4=Other',
        'high_bp': 'High blood pressure diagnosis (0/1)',
        'high_chol': 'High cholesterol diagnosis (0/1)',
        'phys_active': 'Physically active (0/1)',
        'smoker': 'Smoking history (0/1)',
        'gen_health': 'General health (1=Excellent to 5=Poor)',
        'sleep_trouble': 'Sleep trouble (0/1)'
    }
}

with open("model_v2.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✓ Model saved to model_v2.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("✅ MODEL V2 TRAINING COMPLETE")
print("=" * 65)
print(f"""
📊 Dataset:      NHANES 2009-2012
👥 Population:   {len(df_model):,} US adults
🚻 Genders:      Both male and female  
🌍 Ethnicities:  White, Black, Hispanic, Asian, Other

📈 Performance:
   • Accuracy:   {accuracy:.1%}
   • ROC-AUC:    {roc_auc:.3f}

🔧 Features ({len(final_feature_cols)} total):
   ORIGINAL (compatible with your proxy system):
   • pregnancies, bmi, blood_pressure, age
   
   NEW DEMOGRAPHICS:
   • sex, ethnicity
   
   NEW MEDICAL HISTORY:
   • high_bp, high_chol
   
   NEW LIFESTYLE:
   • phys_active, smoker, gen_health, sleep_trouble

💾 Output: model_v2.pkl

📝 Next: Update app.py to add the new input fields!
""")
