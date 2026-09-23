# Dental OPG Tooth Analyzer

Streamlit web app for dental panoramic X-ray (OPG) tooth detection using YOLOv8.
Detects and labels teeth using the FDI two-digit numbering system.

## Features
- Upload OPG X-ray images (JPG/PNG)
- Bounding boxes drawn with FDI tooth numbers
- Confidence scores per tooth
- Quadrant summary (Q1/Q2/Q3/Q4)
- Missing teeth analysis
- Downloadable PDF report

## Setup

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Copy your model
Copy your trained `best.pt` file into this folder:
```
dental_app/
├── app.py
├── best.pt        ← paste here
├── requirements.txt
└── README.md
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

The app opens automatically in your browser at http://localhost:8501

## Model path
By default the app looks for `best.pt` in the same folder.
If your model is elsewhere, change the path in the sidebar inside the app.

## Download model from Google Drive
If your best.pt is saved in Google Drive from Colab training:
1. Go to Google Drive
2. Find: `dental_runs/opg_v1/weights/best.pt`
3. Right-click → Download
4. Paste into this folder
