import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
from datetime import datetime
from fpdf import FPDF
import tempfile

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dental OPG Analyzer",
    page_icon="🦷",
    layout="wide",
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FDI_NAMES = [
    '11','12','13','14','15','16','17','18',
    '21','22','23','24','25','26','27','28',
    '31','32','33','34','35','36','37','38',
    '41','42','43','44','45','46','47','48',
]

QUADRANT_COLORS = {
    '1': '#E74C3C',   # Q1 — upper right — red
    '2': '#2ECC71',   # Q2 — upper left  — green
    '3': '#3498DB',   # Q3 — lower left  — blue
    '4': '#F39C12',   # Q4 — lower right — orange
}

QUADRANT_LABELS = {
    '1': 'Q1 — Upper Right',
    '2': 'Q2 — Upper Left',
    '3': 'Q3 — Lower Left',
    '4': 'Q4 — Lower Right',
}

QUADRANT_RANGE = {
    '1': '11–18',
    '2': '21–28',
    '3': '31–38',
    '4': '41–48',
}


# ─────────────────────────────────────────────
# Load model (cached so it loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


# ─────────────────────────────────────────────
# Draw bounding boxes on image
# ─────────────────────────────────────────────
def draw_predictions(image: Image.Image, detections: list) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font      = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small= ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font       = ImageFont.load_default()
        font_small = font

    for det in detections:
        fdi             = det['fdi']
        x1, y1, x2, y2 = det['bbox']
        conf            = det['confidence']
        quadrant        = fdi[0]
        color           = QUADRANT_COLORS.get(quadrant, '#FFFFFF')

        # Draw bounding box (thick border)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color
            )

        # Label background
        label     = f"{fdi} ({conf:.0%})"
        bbox_text = draw.textbbox((x1, y1), label, font=font)
        text_w    = bbox_text[2] - bbox_text[0]
        text_h    = bbox_text[3] - bbox_text[1]

        label_y1 = max(0, y1 - text_h - 6)
        draw.rectangle([x1, label_y1, x1 + text_w + 8, y1], fill=color)
        draw.text((x1 + 4, label_y1 + 2), label, fill='white', font=font)

    return img


# ─────────────────────────────────────────────
# Run YOLO prediction
# ─────────────────────────────────────────────
def run_prediction(model, image: Image.Image, conf_thresh: float) -> list:
    img_array = np.array(image.convert("RGB"))
    results   = model.predict(img_array, conf=conf_thresh, iou=0.4, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        if cls_id >= len(FDI_NAMES):
            continue
        fdi  = FDI_NAMES[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
        detections.append({
            'fdi'       : fdi,
            'confidence': conf,
            'bbox'      : [x1, y1, x2, y2],
            'quadrant'  : fdi[0],
        })

    detections.sort(key=lambda d: int(d['fdi']))
    return detections


# ─────────────────────────────────────────────
# Build quadrant summary
# ─────────────────────────────────────────────
def get_quadrant_summary(detections: list) -> dict:
    summary = {'1': [], '2': [], '3': [], '4': []}
    for det in detections:
        q = det['quadrant']
        if q in summary:
            summary[q].append(det['fdi'])
    return summary


# ─────────────────────────────────────────────
# Generate PDF report
# ─────────────────────────────────────────────
def generate_pdf(detections: list, quadrant_summary: dict,
                 annotated_img: Image.Image, patient_name: str) -> bytes:

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 14, "Dental OPG Analysis Report", ln=True)

    # Sub header
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 20)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # Patient info box
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Patient: {patient_name if patient_name else 'Not specified'}", ln=True, fill=True)
    pdf.cell(0, 8, f"Total teeth detected: {len(detections)}", ln=True, fill=True)
    pdf.ln(6)

    # Save annotated image to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        annotated_img.save(tmp.name, 'JPEG', quality=90)
        tmp_path = tmp.name

    # Insert image
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Annotated OPG Image", ln=True)
    img_w = 180
    pdf.image(tmp_path, x=15, w=img_w)
    os.unlink(tmp_path)
    pdf.ln(6)

    # Quadrant summary table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Quadrant Summary", ln=True)
    pdf.ln(2)

    q_colors = {
        '1': (231, 76,  60),
        '2': (46,  204, 113),
        '3': (52,  152, 219),
        '4': (243, 156, 18),
    }

    for q in ['1', '2', '3', '4']:
        teeth = quadrant_summary[q]
        r, g, b = q_colors[q]
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 8, f" {QUADRANT_LABELS[q]}", fill=True)
        pdf.set_fill_color(245, 245, 245)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        teeth_str = ', '.join(teeth) if teeth else 'None detected'
        pdf.cell(0, 8, f"  {teeth_str}  ({len(teeth)} teeth)", fill=True, ln=True)
        pdf.ln(1)

    pdf.ln(6)

    # Detailed detections table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detailed Detection Results", ln=True)
    pdf.ln(2)

    # Table header
    pdf.set_fill_color(41, 128, 185)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(30,  8, "FDI No.",   border=0, fill=True)
    pdf.cell(40,  8, "Quadrant",  border=0, fill=True)
    pdf.cell(40,  8, "Confidence",border=0, fill=True)
    pdf.cell(0,   8, "Bounding Box",border=0, fill=True, ln=True)

    # Table rows
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    for i, det in enumerate(detections):
        fill = i % 2 == 0
        pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
        x1, y1, x2, y2 = det['bbox']
        pdf.cell(30,  7, f"  {det['fdi']}",      fill=fill)
        pdf.cell(40,  7, f"  {QUADRANT_LABELS[det['quadrant']]}",  fill=fill)
        pdf.cell(40,  7, f"  {det['confidence']:.1%}",             fill=fill)
        pdf.cell(0,   7, f"  [{x1}, {y1}, {x2}, {y2}]",           fill=fill, ln=True)

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6,
             "This report is generated by an AI model and should be reviewed by a qualified dental professional.",
             ln=True)

    return pdf.output(dest='S').encode('latin-1')


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/280x60/2980B9/FFFFFF?text=Dental+OPG+Analyzer",
             use_column_width=True)
    st.markdown("---")

    st.subheader("Model Settings")
    model_path = st.text_input(
        "Path to best.pt",
        value="best.pt",
        help="Full path to your trained YOLOv8 best.pt file"
    )
    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.1, max_value=0.9,
        value=0.4, step=0.05,
        help="Lower = detects more teeth (may include false positives)"
    )

    st.markdown("---")
    st.subheader("Patient Info (optional)")
    patient_name = st.text_input("Patient name", placeholder="e.g. John Doe")
    patient_id   = st.text_input("Patient ID",   placeholder="e.g. PT-00123")

    st.markdown("---")
    st.markdown("**FDI Quadrant Guide**")
    for q in ['1', '2', '3', '4']:
        color = QUADRANT_COLORS[q]
        st.markdown(
            f"<span style='color:{color}; font-weight:600;'>■</span> "
            f"{QUADRANT_LABELS[q]} ({QUADRANT_RANGE[q]})",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.title("🦷 Dental OPG Tooth Analyzer")
st.markdown("Upload a panoramic X-ray (OPG) image to detect and label teeth using the FDI two-digit numbering system.")
st.markdown("---")

# Load model
if not os.path.exists(model_path):
    st.error(f"Model not found at: `{model_path}`\n\nPlease set the correct path in the sidebar.")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model(model_path)
st.success("Model loaded successfully!", icon="✅")

st.markdown("---")

# Upload
uploaded_file = st.file_uploader(
    "Upload OPG X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # Run prediction
    with st.spinner("Analyzing OPG image..."):
        detections       = run_prediction(model, image, conf_thresh)
        annotated_img    = draw_predictions(image, detections)
        quadrant_summary = get_quadrant_summary(detections)

    with col2:
        st.subheader("Detected Teeth")
        st.image(annotated_img, use_column_width=True)

    st.markdown("---")

    # ── Summary metrics row ──
    st.subheader("Detection Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total teeth",  len(detections))
    m2.metric("Q1 (upper R)", len(quadrant_summary['1']))
    m3.metric("Q2 (upper L)", len(quadrant_summary['2']))
    m4.metric("Q3 (lower L)", len(quadrant_summary['3']))
    m5.metric("Q4 (lower R)", len(quadrant_summary['4']))

    st.markdown("---")

    # ── Quadrant breakdown ──
    st.subheader("Quadrant Breakdown")
    qc1, qc2, qc3, qc4 = st.columns(4)

    for col, q in zip([qc1, qc2, qc3, qc4], ['1', '2', '3', '4']):
        teeth = quadrant_summary[q]
        color = QUADRANT_COLORS[q]
        with col:
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 10px;'>"
                f"<b style='color:{color}'>{QUADRANT_LABELS[q]}</b><br>"
                f"<small>Teeth {QUADRANT_RANGE[q]}</small>"
                f"</div>",
                unsafe_allow_html=True
            )
            if teeth:
                for t in teeth:
                    st.markdown(f"&nbsp;&nbsp;• Tooth **{t}**")
            else:
                st.markdown("*None detected*")

    st.markdown("---")

    # ── Full detections table ──
    st.subheader("All Detections")

    if detections:
        import pandas as pd
        df = pd.DataFrame([{
            'FDI Number' : det['fdi'],
            'Quadrant'   : QUADRANT_LABELS[det['quadrant']],
            'Confidence' : f"{det['confidence']:.1%}",
            'BBox x1'    : det['bbox'][0],
            'BBox y1'    : det['bbox'][1],
            'BBox x2'    : det['bbox'][2],
            'BBox y2'    : det['bbox'][3],
        } for det in detections])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No teeth detected. Try lowering the confidence threshold in the sidebar.")

    st.markdown("---")

    # ── Missing teeth ──
    st.subheader("Missing Teeth Analysis")
    detected_fdis = {det['fdi'] for det in detections}
    missing       = [t for t in FDI_NAMES if t not in detected_fdis]

    if missing:
        miss_by_q = {'1': [], '2': [], '3': [], '4': []}
        for t in missing:
            miss_by_q[t[0]].append(t)

        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, q in zip([mc1, mc2, mc3, mc4], ['1', '2', '3', '4']):
            with col:
                color = QUADRANT_COLORS[q]
                st.markdown(
                    f"<small style='color:{color}'><b>{QUADRANT_LABELS[q]}</b></small>",
                    unsafe_allow_html=True
                )
                if miss_by_q[q]:
                    st.markdown(", ".join(miss_by_q[q]))
                else:
                    st.markdown("*All present*")
    else:
        st.success("All 32 teeth detected!")

    st.markdown("---")

    # ── PDF Download ──
    st.subheader("Download Report")
    with st.spinner("Generating PDF report..."):
        try:
            pdf_bytes = generate_pdf(
                detections, quadrant_summary,
                annotated_img, patient_name
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname     = f"OPG_Report_{patient_name.replace(' ','_') or 'Patient'}_{timestamp}.pdf"

            st.download_button(
                label     = "📄 Download PDF Report",
                data      = pdf_bytes,
                file_name = fname,
                mime      = "application/pdf",
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}\nMake sure fpdf2 is installed: pip install fpdf2")

else:
    # Empty state
    st.info("👆 Upload an OPG X-ray image above to begin analysis.")
    st.markdown("""
    **How to use:**
    1. Set the path to your `best.pt` model in the sidebar
    2. Optionally enter patient name and ID
    3. Upload an OPG panoramic X-ray image
    4. View detected teeth with FDI numbers, confidence scores, and quadrant breakdown
    5. Download the PDF report
    """)
