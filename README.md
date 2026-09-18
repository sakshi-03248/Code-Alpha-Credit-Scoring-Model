# Credit Scoring Model

## CodeAlpha Internship Project

A machine learning classification project that predicts the credit risk of loan applicants using personal, employment, and loan-related information.

## Project Objectives

- Explore and understand the credit-risk dataset.
- Clean and preprocess the data.
- Handle missing values.
- Encode categorical variables.
- Train multiple classification models.
- Compare model performance.
- Tune the best-performing model.
- Evaluate the final model.
- Analyze feature importance.
- Save the trained model for future predictions.

## Dataset

The dataset contains applicant and loan information such as:

- Applicant age
- Applicant income
- Employment length
- Home ownership
- Loan amount
- Loan intent
- Loan interest rate
- Loan grade
- Loan percentage of income
- Previous loan default information
- Credit history length

The target variable represents the applicant's credit-risk class.

## Technologies Used

- Python
- Google Colab
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Models Trained

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Random Forest was selected as the final model because it produced the strongest results among the tested models.

## Model Comparison

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Random Forest Classifier | 93.42% | 0.9307 |
| Logistic Regression | 87.16% | 0.8766 |
| Decision Tree Classifier | 89.01% | 0.8494 |

## Final Model Performance

The tuned Random Forest model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 93.39% |
| Precision | 93.03% |
| Recall | 93.39% |
| F1-Score | 93.01% |
| ROC-AUC | 0.9319 |

These results are based on the test dataset used in the notebook.

## Important Features

The most important features included:

1. Loan-to-income ratio
2. Loan percentage of income
3. Applicant income
4. Loan interest rate
5. Home ownership
6. Loan grade
7. Loan amount
8. Employment length
9. Applicant age

Feature importance indicates which variables the model relies on most. It does not prove causation.

## Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
│
├── data/
│   └── Dataset files
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── models/
│   └── credit_scoring_model.pkl
├── notebooks/
│   └── Credit_Scoring_Model_CodeAlpha.ipynb
├── src/
│   └── prediction.py
├── README.md
└── requirements.txt
```

## How to Run in Google Colab

### 1. Open Google Colab

Visit https://colab.research.google.com/

### 2. Upload the Notebook

- Click **File**
- Select **Upload notebook**
- Upload `Credit_Scoring_Model_CodeAlpha.ipynb`

### 3. Upload the Dataset

If the dataset is not available, run:

```python
from google.colab import files
uploaded = files.upload()
```

Select the dataset file.

### 4. Check the Dataset Path

Make sure the filename matches the code in the notebook. For example:

```python
df = pd.read_csv("credit_risk_dataset.csv")
```

Change the filename if necessary.

### 5. Install Dependencies if Required

```python
!pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### 6. Run the Notebook

Select:

```text
Runtime → Run all
```

Or execute the cells from top to bottom.

### 7. Review the Results

Review the data analysis, model comparison, final metrics, confusion matrix, ROC curve, and feature-importance chart.

## How to Run Locally

1. Install Python 3.9 or newer.
2. Open a terminal inside the project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open:

```text
notebooks/Credit_Scoring_Model_CodeAlpha.ipynb
```

6. Place the dataset inside the `data/` folder.
7. Update the dataset path if necessary.
8. Run all notebook cells.

## Loading the Saved Model

```python
import joblib

model = joblib.load("models/credit_scoring_model.pkl")
```

The prediction data must use the same preprocessing, feature columns, encoding, and feature order used during training.

## Evaluation Metrics

- **Accuracy:** Percentage of correct predictions.
- **Precision:** Percentage of predicted positive cases that are actually positive.
- **Recall:** Percentage of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures class-separation performance across thresholds.
- **Confusion matrix:** Shows true positives, true negatives, false positives, and false negatives.

## Limitations

- Results depend on the quality and representativeness of the dataset.
- Performance may change on new data.
- The model has not been deployed in a production lending system.
- Additional fairness, bias, and external-validation testing is required before real-world use.
- Credit decisions should not rely only on this model.

## Future Improvements

- Test additional algorithms.
- Apply advanced hyperparameter optimization.
- Add SHAP or LIME explainability.
- Build a Streamlit web interface.
- Deploy using Flask or FastAPI.
- Add fairness and bias analysis.
- Validate on external datasets.
- Add model monitoring and automated data validation.

## Conclusion

This project demonstrates the use of machine learning for credit-risk classification. Multiple classification models were trained and compared, and Random Forest achieved the strongest performance among the tested models.

The project provides a foundation for a more advanced credit scoring system with additional validation, explainability, fairness testing, and deployment.
