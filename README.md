# Heart Disease Prediction

A logistic regression model that predicts the presence of heart disease from 13 clinical features, with a Streamlit app for interactive predictions.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis.

**Live demo:** _add your Streamlit Community Cloud link here_

![App screenshot](assets/app_screenshot.png)

## Results

Evaluated on a stratified 80/20 hold-out split (61 test samples, `random_state=42`).

| Metric | Value |
|---|---|
| Accuracy | 0.885 |
| Precision (disease) | 0.91 |
| Recall (disease) | 0.88 |
| F1 (disease) | 0.89 |
| ROC-AUC | _fill in from notebook_ |
| 5-fold CV accuracy | _mean ± std from notebook_ |

Confusion matrix (hold-out):

|  | Predicted healthy | Predicted disease |
|---|---|---|
| **Actual healthy** | 25 | 3 |
| **Actual disease** | 4 | 29 |

In a screening context, false negatives (4 missed cases) cost more than false positives, so recall on the disease class is the metric to watch. See the notebook for a threshold analysis.

_Update the table with the numbers from your re-run notebook; the values above come from the original run._

## Dataset

The UCI Heart Disease (Cleveland) dataset: 303 patients, 13 features, binary target (`1` = disease present). Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG result (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0-3) |
| `thal` | Thalassemia result |

## Approach

1. Exploratory analysis: class balance, correlations, feature distributions.
2. Preprocessing: `StandardScaler` inside a scikit-learn `Pipeline`, so the app applies identical scaling at inference.
3. Model: Logistic Regression (`max_iter=1000`), compared against Random Forest and SVM.
4. Evaluation: stratified split, 5-fold cross-validation, confusion matrix, ROC-AUC.
5. Interpretation: model coefficients to show which features drive predictions.

## Project structure

```
.
├── app.py                      # Streamlit app
├── data/heart.csv
├── notebooks/Logistic_Regression.ipynb
├── models/logistic_model.pkl   # full pipeline (scaler + model)
├── assets/app_screenshot.png
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/maheshwarikalaskar/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn, joblib, Streamlit, matplotlib

## Future work

- Probability calibration and decision-threshold tuning for higher recall
- SHAP-based explanations in the app
- Validation on an external dataset

## Author

**Maheshwari Kalaskar**: [GitHub](https://github.com/maheshwarikalaskar) · [LinkedIn](https://www.linkedin.com/in/maheshwarikalaskar/)
