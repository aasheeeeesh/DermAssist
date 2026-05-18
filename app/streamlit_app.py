"""
DermAssist — Explainable Skin Lesion Classification
====================================================
Phase 4: Streamlit Web Application

Features:
  - Upload any dermoscopy image
  - Real-time ResNet50 feature extraction
  - SVM / all-model predictions with confidence
  - Captum Occlusion XAI heatmap overlay
  - Model performance comparison charts
"""

import sys
import io
import time
from pathlib import Path

import numpy as np
import torch
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import paths, preprocessing_config, model_config, app_config
from src.embeddings import get_feature_extractor
from src.interpretability import generate_occlusion_heatmap

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermAssist — Skin Lesion AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS — dark premium design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Page background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #1c2128 100%) !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
}

/* Header card */
.hero-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #7c3aed, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* Metric cards */
.metric-card {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(88,166,255,0.15);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-label {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Result boxes */
.result-benign {
    background: linear-gradient(135deg, #0f2722, #0d2016);
    border: 1px solid #238636;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-malignant {
    background: linear-gradient(135deg, #2d1313, #1c0d0d);
    border: 1px solid #da3633;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-label {
    font-size: 1.8rem;
    font-weight: 700;
}
.result-conf {
    font-size: 1rem;
    color: #8b949e;
    margin-top: 0.3rem;
}

/* Progress bar override */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #58a6ff, #7c3aed) !important;
    border-radius: 8px !important;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #30363d;
    margin: 1.5rem 0;
}

/* Warning / info boxes */
.info-box {
    background: #1c2128;
    border: 1px solid #1f6feb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #79c0ff;
    font-size: 0.9rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    background: #161b22 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(88,166,255,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    model, dim = get_feature_extractor("resnet50")
    return model, dim

@st.cache_resource(show_spinner=False)
def load_classifier(model_path):
    return joblib.load(model_path)

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Resize, normalise and convert a PIL image to a model-ready tensor."""
    transform = transforms.Compose([
        transforms.Resize(preprocessing_config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocessing_config.NORMALIZE_MEAN,
            std=preprocessing_config.NORMALIZE_STD,
        ),
    ])
    return transform(pil_img.convert("RGB"))

def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Denormalise a tensor back to displayable uint8 numpy array."""
    mean = torch.tensor(preprocessing_config.NORMALIZE_MEAN).view(3,1,1)
    std  = torch.tensor(preprocessing_config.NORMALIZE_STD).view(3,1,1)
    img  = tensor * std + mean
    img  = img.clamp(0, 1).permute(1,2,0).numpy()
    return (img * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  HEATMAP OVERLAY RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_heatmap_overlay(original_tensor, attributions):
    """Blends a semi-transparent attribution heatmap over the original image."""
    orig_np  = tensor_to_display(original_tensor)
    attr_np  = attributions.permute(1,2,0).numpy()
    heat_raw = np.abs(attr_np).sum(axis=-1)

    # Normalise 0..1
    if heat_raw.max() > 0:
        heat_norm = (heat_raw - heat_raw.min()) / (heat_raw.max() - heat_raw.min())
    else:
        heat_norm = heat_raw

    # Colour map: cool = low importance, hot = high importance
    colormap  = cm.get_cmap("RdYlGn_r")
    heat_rgba = colormap(heat_norm)           # shape (H,W,4)
    heat_rgb  = (heat_rgba[:,:,:3] * 255).astype(np.uint8)

    # Alpha blend
    alpha  = 0.5
    blend  = (alpha * heat_rgb + (1 - alpha) * orig_np).astype(np.uint8)
    return orig_np, heat_norm, blend

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 DermAssist")
    st.markdown("**Explainable AI for Skin Lesion Classification**")
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    enable_xai = st.toggle("Generate XAI Heatmap", value=True)
    show_all_models = st.toggle("Show All Model Predictions", value=False)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    st.markdown("### 📂 Model")
    model_path = paths.MODELS_DIR / "best_classifier.joblib"
    if model_path.exists():
        st.success(f"✅ Model loaded")
        st.caption(f"`best_classifier.joblib`")
    else:
        st.error("❌ No trained model found.\nRun `run_pipeline.py` first.")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    st.markdown("### 📊 Pipeline Metrics")
    st.markdown("""
| Model | Test AUC |
|---|---|
| **SVM** 🏆 | 0.9504 |
| LDA | 0.9424 |
| Random Forest | 0.9373 |
| AdaBoost | 0.9300 |
| Logistic Reg. | 0.9180 |
""")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.caption("Built with ResNet50 + SVM · ISIC Dataset · Captum XAI")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <p class="hero-title">🔬 DermAssist</p>
  <p class="hero-sub">
    Explainable AI · ResNet50 Feature Extraction · SVM Classification · Captum XAI Heatmaps<br>
    Trained on <strong>700 real ISIC dermoscopy images</strong> · Test AUC = <strong>0.95</strong>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Top metrics row ────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-value">95%</div><div class="metric-label">AUC-ROC (SVM)</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-value">88.6%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-value">2048</div><div class="metric-label">Embedding Dims</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">Models Compared</div></div>', unsafe_allow_html=True)

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CONTENT — TWO COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

# ── LEFT: Upload & Result ─────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">📤 Upload Dermoscopy Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a skin lesion image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if not model_path.exists():
        st.error("⚠️ No trained model found. Please run `run_pipeline.py` first.")
        st.stop()

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        analyse_btn = st.button("🔍  Analyse Image", use_container_width=True)

        if analyse_btn:
            # ── Load models ───────────────────────────────────────────────
            with st.spinner("Loading ResNet50 feature extractor..."):
                feature_extractor, _ = load_feature_extractor()
            with st.spinner("Loading SVM classifier..."):
                classifier = load_classifier(model_path)

            # ── Preprocess ────────────────────────────────────────────────
            img_tensor = preprocess_image(pil_img)

            # ── Extract features & predict ────────────────────────────────
            with st.spinner("Extracting deep features via ResNet50..."):
                device = torch.device(model_config.DEVICE)
                with torch.no_grad():
                    feat = feature_extractor(img_tensor.unsqueeze(0).to(device))
                    feat_np = feat.cpu().numpy()

            with st.spinner("Running SVM classifier..."):
                pred_label = classifier.predict(feat_np)[0]
                pred_proba = classifier.predict_proba(feat_np)[0]
                confidence = pred_proba[pred_label] * 100

            # ── Store results in session ─────────────────────────────────
            st.session_state["pred_label"]   = pred_label
            st.session_state["pred_proba"]   = pred_proba
            st.session_state["confidence"]   = confidence
            st.session_state["img_tensor"]   = img_tensor
            st.session_state["feat_np"]      = feat_np
            st.session_state["feature_extractor"] = feature_extractor
            st.session_state["classifier"]   = classifier
            st.session_state["pil_img"]      = pil_img
            st.session_state["analysed"]     = True

    # ── Show result if analysed ────────────────────────────────────────────
    if st.session_state.get("analysed"):
        pred_label = st.session_state["pred_label"]
        confidence = st.session_state["confidence"]
        pred_proba = st.session_state["pred_proba"]

        st.markdown("---")
        st.markdown('<div class="section-title">🧬 Diagnosis Result</div>', unsafe_allow_html=True)

        if pred_label == 0:
            st.markdown(f"""
<div class="result-benign">
  <div class="result-label" style="color:#3fb950;">✅ BENIGN</div>
  <div class="result-conf">Confidence: {confidence:.1f}%</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-malignant">
  <div class="result-label" style="color:#f85149;">⚠️ MALIGNANT</div>
  <div class="result-conf">Confidence: {confidence:.1f}%</div>
</div>""", unsafe_allow_html=True)

        # Probability bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probability Breakdown**")
        st.markdown(f"🟢 Benign: **{pred_proba[0]*100:.1f}%**")
        st.progress(float(pred_proba[0]))
        st.markdown(f"🔴 Malignant: **{pred_proba[1]*100:.1f}%**")
        st.progress(float(pred_proba[1]))

        # Disclaimer
        st.markdown("""
<div class="info-box">
  ⚕️ <strong>Clinical Disclaimer</strong>: This tool is for educational/research use only.
  Always consult a certified dermatologist for medical diagnosis.
</div>""", unsafe_allow_html=True)

# ── RIGHT: XAI Heatmap & Charts ──────────────────────────────────────────────
with col_right:
    if st.session_state.get("analysed"):
        img_tensor       = st.session_state["img_tensor"]
        feature_extractor = st.session_state["feature_extractor"]
        classifier       = st.session_state["classifier"]
        pred_label       = st.session_state["pred_label"]

        # ── XAI Heatmap ─────────────────────────────────────────────────
        if enable_xai:
            st.markdown('<div class="section-title">🧠 XAI Explainability Heatmap</div>', unsafe_allow_html=True)

            with st.spinner("Generating occlusion heatmap (this takes ~30s on CPU)..."):
                try:
                    attributions = generate_occlusion_heatmap(
                        img_tensor, feature_extractor, classifier,
                        target_class=int(pred_label)
                    )
                    orig_np, heat_norm, blend = render_heatmap_overlay(img_tensor, attributions)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0d1117")
                    titles = ["Original Image", "Attribution Map", "Overlay (Important Regions)"]
                    imgs   = [orig_np, heat_norm, blend]
                    cmaps  = [None, "RdYlGn_r", None]

                    for ax, title, data, cmap in zip(axes, titles, imgs, cmaps):
                        ax.imshow(data, cmap=cmap)
                        ax.set_title(title, color="#e6edf3", fontsize=10, pad=8)
                        ax.axis("off")
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                    # Colorbar for heatmap
                    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                               norm=plt.Normalize(vmin=0, vmax=1))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
                    cbar.ax.set_ylabel("Attribution Intensity", color="#8b949e", fontsize=8)
                    cbar.ax.tick_params(colors="#8b949e")

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    st.markdown("""
<div class="info-box">
  🔴 <strong>Red / Dark regions</strong>: Highly influential — covering these pixels dramatically changes the prediction.<br>
  🟢 <strong>Green / Light regions</strong>: Low influence on the model's decision.
</div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.warning(f"Heatmap generation skipped: {e}")

        # ── Model Performance Comparison ────────────────────────────────
        st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

        model_names = ["SVM", "Logistic Reg.", "Random Forest", "AdaBoost", "LDA"]
        test_acc  = [0.8857, 0.8667, 0.8762, 0.8667, 0.8762]
        test_f1   = [0.8667, 0.8444, 0.8602, 0.8571, 0.8660]
        test_auc  = [0.9504, 0.9180, 0.9373, 0.9300, 0.9424]

        fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor="#161b22")
        x   = np.arange(len(model_names))
        w   = 0.25
        ax2.set_facecolor("#161b22")

        b1 = ax2.bar(x - w,  test_acc, w, label="Test Accuracy", color="#58a6ff", alpha=0.9)
        b2 = ax2.bar(x,      test_f1,  w, label="Test F1-Score", color="#bc8cff", alpha=0.9)
        b3 = ax2.bar(x + w,  test_auc, w, label="Test AUC-ROC",  color="#3fb950", alpha=0.9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, color="#e6edf3", fontsize=9)
        ax2.set_ylim(0.75, 1.02)
        ax2.set_ylabel("Score", color="#8b949e")
        ax2.set_title("All 5 Classifiers — Real ISIC Dataset (700 images)",
                      color="#e6edf3", fontsize=11, pad=10)
        ax2.tick_params(colors="#8b949e")
        ax2.spines[["bottom","left"]].set_color("#30363d")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.yaxis.grid(True, color="#21262d", linestyle="--", alpha=0.7)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=9, labelcolor="#e6edf3", facecolor="#1c2128",
                   edgecolor="#30363d")

        for bar in [*b1, *b2, *b3]:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                     f"{h:.2f}", ha="center", va="bottom",
                     fontsize=7, color="#8b949e")

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    else:
        # ── Placeholder when no image uploaded ───────────────────────────
        st.markdown('<div class="section-title">🧠 XAI Explainability Heatmap</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="background:#161b22; border:1px dashed #30363d; border-radius:12px;
            padding: 3rem; text-align:center; color:#8b949e;">
  <span style="font-size:3rem;">🔬</span><br><br>
  <strong style="color:#e6edf3;">Upload an image and click Analyse</strong><br>
  <small>The XAI heatmap will appear here, showing which skin regions the AI focused on.</small>
</div>""", unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
```
📷 Input Image (224×224)
       ↓
🧠 ResNet50 (pretrained ImageNet)
   — removes final FC layer →
       ↓
📐 2048-dim Feature Embedding
       ↓
🤖 SVM Classifier (RBF kernel)
   — trained on 490 real ISIC images →
       ↓
🎯 Prediction: Benign / Malignant
       ↓
🔥 Captum Occlusion Heatmap (XAI)
```
""")

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:0.85rem; padding:0.5rem 0;">
  DermAssist · Built with PyTorch, Scikit-Learn, Captum & Streamlit ·
  ISIC Dataset (CC0) · For Research & Educational Use Only
</div>
""", unsafe_allow_html=True)
