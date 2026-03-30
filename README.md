<img src="cover.png" width="100%">

# AI-Powered Defect Logging System (Web-based, Mobile & Desktop)

## 1. Overview

This project is a web-based defect classification workflow for manufacturing surface defects.
A user can train a lightweight image classification model, run single-image inference, and open a Streamlit web interface that works on both desktop and mobile browsers.

The current validated workflow is centered on three executable scripts:

- `train_neu_from_train_validation.py`
- `predict_one_neu.py`
- `app.py`

The repository is designed around the NEU Surface Defect dataset folder structure using `train/` and `validation/` directories.

## 2. Key Features

- MobileNetV2-based transfer learning for defect classification
- Single-image prediction script for quick verification
- Streamlit-based web interface for desktop and mobile usage
- Class index export for consistent label mapping
- Ready-to-upload GitHub repository structure
- Apache License 2.0 package included

## 3. Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pandas
- Pillow

## 4. Dataset Structure

Place your dataset like this:

```text
train/
├─ crazing/
├─ inclusion/
├─ patches/
├─ pitted_surface/
├─ rolled-in_scale/
└─ scratches/

validation/
├─ crazing/
├─ inclusion/
├─ patches/
├─ pitted_surface/
├─ rolled-in_scale/
└─ scratches/
```

## 5. Main Scripts

### `train_neu_from_train_validation.py`
Trains a MobileNetV2-based classifier from `train/` and `validation/` folders and saves:

- `mobilenetv2_neu_defect_best.keras`
- `class_indices.json`

### `predict_one_neu.py`
Runs single-image inference using the saved model and exported class mapping.

### `app.py`
Launches a Streamlit interface for image upload or camera input and displays prediction results.

## 6. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_neu_from_train_validation.py
```

Run single-image prediction:

```bash
python predict_one_neu.py sample.jpg
```

Launch the web app:

```bash
streamlit run app.py
```

## 7. Output Files

Training generates:

- `mobilenetv2_neu_defect_best.keras`
- `class_indices.json`

The Streamlit app may also create local folders such as:

- `uploads/`

## 8. Repository Structure

```text
ai_defect_logging_system_web_mobile/
├─ app.py
├─ predict_one_neu.py
├─ train_neu_from_train_validation.py
├─ README.md
├─ LICENSE
├─ NOTICE
├─ requirements.txt
├─ environment.yml
├─ pyproject.toml
├─ MANIFEST.in
├─ .gitignore
├─ CHANGELOG.md
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ SECURITY.md
├─ run_train.bat
├─ run_predict.bat
├─ run_streamlit.bat
├─ docs/
│  ├─ PROJECT_OVERVIEW.md
│  ├─ USAGE.md
│  └─ ROADMAP.md
├─ data/
│  ├─ train/
│  └─ validation/
├─ models/
└─ uploads/
```

## 9. Current Scope

This repository currently focuses on:

- training
- inference
- web-based classification UI

Persistent production logging, database-backed record management, and analytics dashboards can be expanded in future versions.

## 10. Roadmap

Planned extensions:

- structured result logging
- record search and filtering
- dashboard analytics
- defect image history management
- cloud deployment
- object detection for defect localization

## 11. License

Apache License 2.0

See `LICENSE` and `NOTICE` for details.

## 12. Author

Yirang Jung
