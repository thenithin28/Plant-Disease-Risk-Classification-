import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom AgriTech Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #022C22 0%, #064E3B 50%, #14532D 100%);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(52, 211, 153, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(52, 211, 153, 0.35);
        margin-bottom: 10px;
    }

    .disease-card {
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .disease-hero {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 6px 0;
    }

    .remedy-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("plant_disease_classifier.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Disease Metadata
DISEASE_META = {
    "Healthy": {
        "color": "#10B981",
        "bg": "linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.06) 100%)",
        "border": "rgba(16, 185, 129, 0.45)",
        "icon": "🌱",
        "desc": "Foliage exhibits vibrant chlorophyll pigmentation, intact cuticle texture, and minimal necrosis.",
        "treatment": "Maintain current irrigation, nutrient schedule, and pest scouting routine."
    },
    "Powdery Mildew": {
        "color": "#38BDF8",
        "bg": "linear-gradient(135deg, rgba(56, 189, 248, 0.16) 0%, rgba(2, 132, 199, 0.06) 100%)",
        "border": "rgba(56, 189, 248, 0.45)",
        "icon": "🍄",
        "desc": "Fungal mycelium growth fueled by high canopy humidity (70%+) and moderate temperatures.",
        "treatment": "Increase canopy airflow, reduce overhead watering, and apply sulfur or bio-fungicide sprays."
    },
    "Leaf Spot": {
        "color": "#F59E0B",
        "bg": "linear-gradient(135deg, rgba(245, 158, 11, 0.16) 0%, rgba(217, 119, 6, 0.06) 100%)",
        "border": "rgba(245, 158, 11, 0.45)",
        "icon": "🍂",
        "desc": "Necrotic lesions spreading across leaf surface with significant spot clustering (20+ spots).",
        "treatment": "Prune infected lower foliage, avoid leaf wetting during watering, and treat with copper-based bactericide/fungicide."
    },
    "Rust": {
        "color": "#F97316",
        "bg": "linear-gradient(135deg, rgba(249, 115, 22, 0.16) 0%, rgba(194, 65, 12, 0.06) 100%)",
        "border": "rgba(249, 115, 22, 0.45)",
        "icon": "🍁",
        "desc": "Pucciniales fungal pustules causing chlorosis, elevated surface texture roughness, and moisture stress.",
        "treatment": "Quarantine affected plants, eliminate standing water, and apply triazole-based protective systemic fungicide."
    }
}

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">AgriTech & Plant Pathology Diagnosis AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">🌿 Plant Disease Classification</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Diagnose crop foliar pathologies (Healthy, Powdery Mildew, Leaf Spot, Rust) using leaf biometrics, lesion counts, moisture sensors, and plant age.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Foliage Pathology Diagnostic", "📁 Crop Field Survey (CSV)", "📊 Model Performance & Confusion Matrix"])

# --- TAB 1: Foliage Pathology Diagnostic ---
with tabs[0]:
    st.subheader("Crop Biometrics & Environmental Telemetry")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 1, 2])
    with p_cols[0]:
        load_h = st.button("🌱 Vibrant Healthy Crop", width="stretch")
    with p_cols[1]:
        load_pm = st.button("🍄 Powdery Mildew", width="stretch")
    with p_cols[2]:
        load_ls = st.button("🍂 Leaf Spot Lesions", width="stretch")
    with p_cols[3]:
        load_r = st.button("🍁 Rust Fungus", width="stretch")

    if load_h:
        st.session_state["color"] = 8.2
        st.session_state["spots"] = 2
        st.session_state["moisture"] = 58.0
        st.session_state["texture"] = 8.2
        st.session_state["age"] = 65
        st.session_state["soil"] = 55.0
        st.session_state["temp"] = 25.0
    elif load_pm:
        st.session_state["color"] = 5.8
        st.session_state["spots"] = 12
        st.session_state["moisture"] = 72.0
        st.session_state["texture"] = 5.0
        st.session_state["age"] = 70
        st.session_state["soil"] = 68.0
        st.session_state["temp"] = 23.0
    elif load_ls:
        st.session_state["color"] = 4.8
        st.session_state["spots"] = 25
        st.session_state["moisture"] = 48.0
        st.session_state["texture"] = 4.8
        st.session_state["age"] = 75
        st.session_state["soil"] = 45.0
        st.session_state["temp"] = 27.0
    elif load_r:
        st.session_state["color"] = 5.4
        st.session_state["spots"] = 18
        st.session_state["moisture"] = 62.0
        st.session_state["texture"] = 5.5
        st.session_state["age"] = 80
        st.session_state["soil"] = 60.0
        st.session_state["temp"] = 29.0

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 🍃 Leaf Morphological Biometrics")
        color_score = st.slider(
            "Leaf Color Score (1 = Chlorotic / Yellow, 10 = Deep Green)", 1.0, 10.0,
            value=float(st.session_state.get("color", 5.2)), step=0.1,
            key="input_color"
        )
        spot_count = st.slider(
            "Leaf Spot / Lesion Count", 0, 45,
            value=int(st.session_state.get("spots", 20)), step=1,
            key="input_spots", help="Total necrotic fungal/bacterial spot blemishes observed."
        )
        texture_score = st.slider(
            "Leaf Texture Score (1 = Blistered / Brittle, 10 = Smooth / Elastic)", 1.0, 10.0,
            value=float(st.session_state.get("texture", 5.3)), step=0.1,
            key="input_texture"
        )
        leaf_moisture = st.slider(
            "Leaf Surface Moisture (%)", 10.0, 100.0,
            value=float(st.session_state.get("moisture", 61.0)), step=1.0,
            key="input_leaf_moisture"
        )

    with c_right:
        st.markdown("#### 🌦️ Growing Environment & Plant Maturity")
        plant_age = st.slider(
            "Plant Age (Days Post-Emergence)", 5, 150,
            value=int(st.session_state.get("age", 70)), step=1,
            key="input_age"
        )
        soil_moisture = st.slider(
            "Rootzone Soil Moisture (%)", 10.0, 100.0,
            value=float(st.session_state.get("soil", 58.0)), step=1.0,
            key="input_soil"
        )
        temperature = st.slider(
            "Ambient Canopy Temperature (°C)", 10.0, 45.0,
            value=float(st.session_state.get("temp", 28.0)), step=0.5,
            key="input_temp"
        )

    st.markdown("---")
    eval_btn = st.button("🚀 Run Pathology Diagnostic", type="primary", width="stretch")

    sample_df = pd.DataFrame([{
        "Leaf_Color_Score": color_score,
        "Leaf_Spot_Count": spot_count,
        "Leaf_Moisture": leaf_moisture,
        "Leaf_Texture_Score": texture_score,
        "Plant_Age_Days": plant_age,
        "Soil_Moisture": soil_moisture,
        "Temperature": temperature
    }])

    pred = model.predict(sample_df)[0]
    probs = model.predict_proba(sample_df)[0]
    classes = list(model.classes_)
    conf = max(probs) * 100
    meta = DISEASE_META.get(pred, DISEASE_META["Healthy"])

    st.markdown("### 📋 Agronomic Pathology Outcome")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        st.markdown(f"""
        <div class="disease-card" style="background: {meta['bg']}; border: 1px solid {meta['border']};">
            <span style="font-size: 2.8rem;">{meta['icon']}</span>
            <div style="color: {meta['color']}; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                Diagnosed Condition
            </div>
            <div class="disease-hero" style="color: {meta['color']};">
                {pred}
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Diagnostic Confidence: <strong>{conf:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Pathology Probability Breakdown")
        for cls_name, p in zip(classes, probs):
            st.write(f"• **{cls_name}:** {p * 100:.1f}%")
            st.progress(float(p))

    with r2:
        st.markdown(f"#### 🔍 Clinical Pathology Notes: {pred}")
        st.info(meta["desc"])

        st.markdown(f"""
        <div class="remedy-box" style="border-left-color: {meta['color']};">
            <strong style="color: {meta['color']};">Recommended Agronomic Action:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {meta['treatment']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Sensor Vector"):
        st.dataframe(sample_df, width="stretch")

# --- TAB 2: Crop Field Survey ---
with tabs[1]:
    st.subheader("Batch Crop Field Pathology Screening")
    st.write("Upload a farm sensor survey CSV or screen against the 720-sample baseline dataset.")

    csv_file = st.file_uploader("Upload Foliage Telemetry CSV", type=["csv"], key="plant_csv")
    df_field = None

    if csv_file is not None:
        df_field = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_field)} field samples from file.")
    else:
        sample_path = get_asset_path("data/plant_disease_data.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline crop health dataset (`data/plant_disease_data.csv`)", value=True):
                df_field = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_field)} samples from baseline dataset.")

    if df_field is not None:
        req_cols = ["Leaf_Color_Score", "Leaf_Spot_Count", "Leaf_Moisture", "Leaf_Texture_Score", "Plant_Age_Days", "Soil_Moisture", "Temperature"]
        missing = [c for c in req_cols if c not in df_field.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Screen Field Survey", type="primary"):
                with st.spinner("Classifying foliage health..."):
                    preds = model.predict(df_field[req_cols])
                    probs = model.predict_proba(df_field[req_cols])

                    res_df = df_field.copy()
                    res_df["Diagnosed_Disease"] = preds
                    res_df["Confidence_%"] = np.round(np.max(probs, axis=1) * 100, 1)

                    healthy_c = sum(preds == "Healthy")
                    disease_c = len(res_df) - healthy_c
                    infection_rate = (disease_c / len(res_df)) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Plants Surveyed", len(res_df))
                    m2.metric("Infected Crops", disease_c, delta=f"{infection_rate:.1f}% infection", delta_color="inverse")
                    m3.metric("Healthy Crops", healthy_c)
                    m4.metric("Avg Confidence", f"{np.mean(np.max(probs, axis=1)) * 100:.1f}%")

                    f_choice = st.selectbox("Filter by Diagnosed Condition:", ["All"] + list(np.unique(preds)))
                    view = res_df
                    if f_choice != "All":
                        view = res_df[res_df["Diagnosed_Disease"] == f_choice]

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Pathology Survey as CSV",
                        data=csv_export,
                        file_name="crop_disease_survey_report.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Performance ---
with tabs[2]:
    st.subheader("Model Architecture & Classification Benchmark")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 Pathology Classifier Specifications
        - **Algorithm**: `RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42)`
        - **Classes Diagnosed**:
            - `Healthy` (Vibrant chlorophyll, minimal spots)
            - `Powdery Mildew` (High moisture & moderate temp fungal infection)
            - `Leaf Spot` (Necrotic spot clustering)
            - `Rust` (Pucciniales fungal rust chlorosis)
        - **Accuracy Benchmark**:
            - **Overall Accuracy**: **99.00%** on Stratified Test Split
            - Near-perfect precision and recall across all four classes.
        """)

    with c2:
        img_path = get_asset_path("plant_disease_confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Confusion Matrix on Test Split", width="stretch")
        else:
            st.info("Confusion matrix image not found.")

st.caption("Precision Agriculture & Plant Pathology AI • Scikit-learn & Streamlit")
