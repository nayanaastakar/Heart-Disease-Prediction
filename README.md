# Heart Disease Prediction

A simple heart disease risk prediction project built with Python and Streamlit.

## Overview

- `front.py` is a Streamlit app for entering patient details and predicting heart disease risk.
- `back.py` trains a Random Forest model from available heart disease dataset files and saves `heart_rf_model.pkl`.
- `heart.csv`, `heart1.csv`, and `Heart_Disease_Prediction.csv` are dataset sources included for experimentation.
- `heart_rf_model.pkl` is the trained model used by the Streamlit app.

## Files

- `back.py` - dataset loading, model training, evaluation, and model export.
- `front.py` - Streamlit user interface and prediction flow.
- `heart.csv` - primary dataset used for training.
- `heart1.csv` - alternate dataset format that can also be loaded automatically.
- `Heart_Disease_Prediction.csv` - additional dataset reference file.
- `heart_rf_model.pkl` - saved trained model used by `front.py`.
- `requirements.txt` - Python dependencies for running and developing the project.

## Requirements

- Python 3.10+
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `streamlit`

## Setup

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Train the model:

```bash
python back.py
```

This will create `heart_rf_model.pkl` in the project folder.

3. Run the Streamlit app:

```bash
streamlit run front.py
```

4. Open the app at:

```text
http://localhost:8501
```

## Notes

- The app uses the trained `heart_rf_model.pkl` model file. If the model file is missing, run `python back.py` first.
- The current UI is designed for educational and demo purposes and is not a medical diagnosis tool.
