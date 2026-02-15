# Machine Learning Assignment 2

## a. Problem Statement

To implement and compare multiple machine learning classification algorithms 
on a real-world dataset and evaluate their performance using standard metrics. 
The models are deployed using Streamlit for interactive evaluation.

---

## b. Dataset Description

Dataset: Breast Cancer Wisconsin Dataset (UCI)

- Total Instances: 569
- Total Features: 30
- Type: Binary Classification
- Target Classes:
    - 0 → Malignant
    - 1 → Benign

---

## c. Models Used

Six classification models were implemented:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

### Evaluation Metrics Used

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Table

| ML Model Name       |   Accuracy |      AUC |   Precision |   Recall |   F1 Score |      MCC |
|:--------------------|-----------:|---------:|------------:|---------:|-----------:|---------:|
| Logistic Regression |   0.982456 | 0.99537  |    0.986111 | 0.986111 |   0.986111 | 0.962302 |
| Decision Tree       |   0.921053 | 0.916336 |    0.956522 | 0.916667 |   0.93617  | 0.83414  |
| KNN                 |   0.973684 | 0.988426 |    0.96     | 1        |   0.979592 | 0.944155 |
| Naive Bayes         |   0.938596 | 0.987765 |    0.945205 | 0.958333 |   0.951724 | 0.867553 |
| Random Forest       |   0.95614  | 0.993056 |    0.958904 | 0.972222 |   0.965517 | 0.905447 |
| XGBoost             |   0.947368 | 0.993386 |    0.945946 | 0.972222 |   0.958904 | 0.886414 |

---

## Observations on Model Performance


| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Demonstrates strong linear decision boundary performance and provides stable results. |
| Decision Tree | Provides interpretable splits but shows slight overfitting compared to ensemble models. |
| KNN | Performs well after scaling but is sensitive to choice of neighbors. |
| Naive Bayes | Assumes feature independence yet provides competitive performance. |
| Random Forest (Ensemble) | Reduces variance and improves stability using multiple decision trees. |
| XGBoost (Ensemble) | Achieves highest generalization due to boosting mechanism and regularization. |


---

## Deployment

The project is deployed using Streamlit Community Cloud.
