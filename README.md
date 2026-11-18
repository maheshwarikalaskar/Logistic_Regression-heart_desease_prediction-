# Heart Disease Prediction

A logistic regression model that predicts a binary heart-disease label from 13 clinical features, with a Streamlit app for interactive predictions.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis.

## Dataset

`heart.csv` is the Cleveland heart-disease data (a small dataset of roughly 300 patient records) with 13 features and a binary `target` column. Class 1 is treated as the positive class throughout. Check the dataset's documentation for exactly how `target` is encoded before interpreting predictions.

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia test result (coded) |

## Results

The original version of this project (unscaled logistic regression, single 80/20 split, 61 test samples) reached 88.5% accuracy, with 25 + 29 correct and 3 + 4 wrong predictions.

`Logistic_Regression.ipynb` is the improved workflow. A test set of 61 rows makes a single accuracy number noisy, so the notebook:

- scales features inside a `Pipeline`, which also fixes the original convergence warning
- uses a stratified split and 5-fold cross-validation
- reports precision, recall, F1, ROC-AUC, a confusion matrix and an ROC curve
- compares Logistic Regression, Random Forest and SVM
- shows the model's coefficients and a decision-threshold analysis

Run the notebook to see the numbers for each of these.

## Run the app

```bash
git clone https://github.com/maheshwarikalaskar/Logistic_Regression-heart_desease_prediction-.git
cd Logistic_Regression-heart_desease_prediction-
pip install -r requirements.txt
streamlit run app.py
```

The app fits the scaler and logistic regression from `heart.csv` when it starts (under a second), so it does not depend on a saved model file. Input limits come from the data, so it only offers values the model has seen.

## Files

```
.
├── app.py                    # Streamlit app
├── heart.csv                 # dataset
├── Logistic_Regression.ipynb # analysis and evaluation
├── requirements.txt
└── README.md
```

The notebook also saves its fitted pipeline as `logistic_model.pkl`. The app does not need it.

## Limitations

- About 300 records from a single source, so results are uncertain and may not generalise.
- A statistical demo. Predictions are not clinically validated.

## Future work

- Probability calibration and decision-threshold tuning
- Model explanations (for example SHAP) in the app
- Validation on an external dataset

## Author

**Maheshwari Kalaskar**: [GitHub](https://github.com/maheshwarikalaskar) · [LinkedIn](https://www.linkedin.com/in/maheshwarikalaskar/)
