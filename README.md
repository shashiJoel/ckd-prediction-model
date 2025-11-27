# Early Detection of Chronic Kidney Disease (CKD)

**Author: Shashi Priya Songa**

---

## Executive Summary

This project develops a machine learning-based system to predict Chronic Kidney Disease (CKD) at an early stage using clinical and laboratory data. The analysis successfully identified Serum Creatinine as the strongest predictor (408% difference between CKD and non-CKD patients) and achieved excellent model performance with Logistic Regression (98.75% accuracy, 100% recall, perfect ROC-AUC, 98.36% F1-Score). Early diagnosis of CKD is critical for preventing progression to end-stage renal disease and reducing long-term complications.

---

## Rationale

**Why should anyone care about this question?**

Chronic Kidney Disease (CKD) is a progressive condition that often develops silently, with symptoms appearing only after significant kidney damage has occurred. By that stage, treatment options are limited, and many patients require dialysis or transplants to survive. Early detection can drastically slow disease progression, reduce healthcare costs, and most importantly, improve patient quality of life.

- This question is deeply personal — my mother is a CKD patient and has been on dialysis for nearly two years. Watching her spend 12 hours every week hooked to a machine, unable to travel or enjoy simple activities, has been heartbreaking. The cruelest part? Looking back at her medical records, the warning signs were there years ago. Slightly elevated creatinine, declining kidney function—all scattered across routine checkups that no one connected into a pattern.

- This project is about building the tool that could have saved her—and can still save millions of others.

- If this model identifies even one patient early, prevents one person from needing dialysis, spares one family from watching their loved one deteriorate—then every hour I've spent on this project will have been worth it.

- That's why this question is important. Not because the machine learning is sophisticated, but because the consequences of not asking it are measured in lost lives, broken families, and preventable suffering—including my own mother's.

---

## Research Question

**What are you trying to answer?**

Can we develop a machine learning model to predict early-stage chronic kidney disease (CKD) using patient demographic, and clinical data enabling timely intervention before irreversible kidney damage occurs?

---

## Data Sources

**What data will you use to answer your question?**

- **Data Source:** The analysis uses publicly available CKD dataset from [UCI Machine Learning Repository - Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
- **Total Samples:** 400 patients
- **Original Features:** 25 (14 numerical, 11 categorical)
- **Target Variable:** CKD classification (ckd vs notckd)
- **Class Distribution:** 62.5% CKD, 37.5% Not CKD (slight imbalance)
- **Missing Values:** 1,002 total (handled through imputation)

---

## Methodology

**What methods are you using to answer the question?**

### **1. Data Preprocessing**

#### **Data Loading & Cleaning:**
- Loaded dataset from CSV with proper handling of missing value indicators ('?', 'nan', 'NaN', '')
- Column name standardization (removed quotes, whitespace cleaning)
- Data type conversion: Converted numerical columns that were read as strings to proper numeric types using `pd.to_numeric()` with error handling
- Removed unique identifier column (id) as it doesn't contribute to prediction
- Renamed columns to descriptive, standardized names for clarity

#### **Missing Value Handling:**
- **Analysis:** Identified 1,002 missing values across 24 out of 25 features
- **Pattern Recognition:** Missing values concentrated in blood cell-related features (Red Blood Cells: 38%, Red Blood Cell Count: 32.75%, White Blood Cell Count: 26.5%)
- **Imputation Strategy:**
  - **Numerical Features:** K-Nearest Neighbors (KNN) imputation with k=5 neighbors
  - **Categorical Features:** Mode imputation (separate imputer per column)
  - **Critical:** All imputation fitted exclusively on training set to prevent data leakage
  - Test set missing values imputed using patterns learned from training data only

#### **Data Leakage Prevention:**
- **Early Train-Test Split:** Performed 80/20 stratified split BEFORE any preprocessing operations
- **Pipeline Approach:** Implemented sklearn Pipeline to ensure all preprocessing steps are fitted on train set only
- **Transform Consistency:** Test set transformed using train-fitted components (imputers, encoders, scalers)
- **Feature Selection:** Feature importance analysis and selection performed on training set only

### **2. Exploratory Data Analysis (EDA)**

#### **Data Inspection:**
- **Structure Analysis:** Examined dataset shape (400 samples, 25 features), data types, memory usage
- **Statistical Summary:** Generated descriptive statistics (mean, median, std, quartiles) for numerical features
- **Duplicate Detection:** Checked for duplicate rows (found 0 duplicates)
- **Target Variable Analysis:** Examined class distribution (62.5% CKD, 37.5% Not CKD)

#### **Missing Value Visualization:**
- Created bar charts showing missing value counts and percentages by feature
- Generated heatmap visualization to identify missing data patterns
- Identified non-random missing patterns suggesting clinical testing protocols

#### **Distribution Analysis:**
- **Numerical Features:** Created distribution plots (histograms) for all 14 numerical features
- **Box Plots:** Visualized numerical features by target variable to identify discriminative features
- **Violin Plots:** Analyzed distribution shapes and density patterns for key features
- **Key Insights:** Identified right-skewed distributions in kidney function markers (serum creatinine, blood urea) indicating disease progression stages

#### **Categorical Features Analysis:**
- Value counts and frequency analysis for all 11 categorical features
- Cross-tabulation with target variable to identify associations
- Visualization of categorical feature distributions by CKD status

#### **Correlation Analysis:**
- Calculated Pearson correlation matrix for numerical features
- Identified highly correlated features (e.g., haemoglobin-PCV-RBCC correlation >0.9)
- Visualized correlation heatmap to guide feature engineering decisions

#### **Outlier Detection:**
- Used Interquartile Range (IQR) method to identify outliers
- Visualized outliers for top discriminative features
- Analyzed outlier patterns in relation to target variable

### **3. Feature Engineering**

#### **Age Grouping:**
- Categorized age into 4 ordinal groups: 0-30 (Young), 31-50 (Middle), 51-70 (Senior), 70+ (Elderly)
- Encoded as numeric (0, 1, 2, 3) for model compatibility

#### **Composite Scores:**
- **Blood Cell Score:** Weighted combination of haemoglobin (50%), packed cell volume (30%), and red blood cell count (20%) - captures anemia severity
- **Kidney Function Score:** Weighted combination of blood urea (40%) and serum creatinine (60%) - strongest predictor combination
- **Electrolyte Balance Score:** Weighted combination of sodium (50%) and potassium (50%) - captures metabolic imbalances
- All scores normalized by maximum values to ensure comparable scales

#### **Ratio Features:**
- **BUN/Creatinine Ratio:** Standard kidney function indicator (blood_urea / serum_creatinine)
- **Hemoglobin/Age Ratio:** Age-adjusted hemoglobin to account for age-related variations
- **Sodium/Potassium Ratio:** Electrolyte balance indicator
- **WBC/RBC Ratio:** Indicates infection/inflammation levels

#### **Risk Factor Count:**
- Cumulative count of clinical risk factors (hypertension, diabetes mellitus, coronary artery disease)
- Binary encoding (yes=1, no=0) summed across risk factors

#### **Categorical Encoding:**
- LabelEncoder applied to all categorical variables
- Encoders fitted on training set only, applied to test set
- Original categorical columns dropped after encoding to prevent StandardScaler errors

#### **Feature Scaling:**
- StandardScaler (Z-score normalization) applied to all numerical features
- Mean centering and unit variance scaling for optimal model performance
- Scaler fitted on training set, applied to test set

### **4. Feature Selection**

#### **Redundancy Removal:**
- Identified and removed normalized versions of original features
- Eliminated highly correlated features (correlation >0.95) to reduce multicollinearity

#### **Feature Importance Analysis:**
- Used Random Forest classifier to calculate feature importance scores
- Analysis performed exclusively on training set to prevent data leakage
- Ranked features by importance scores

#### **Optimal Feature Selection:**
- Selected top 25 features based on importance scores
- Rationale: 400 samples / 25 features = 16 samples per feature (optimal ratio to prevent overfitting)
- Reduced from 58 engineered features to 25 optimal features

### **5. Predictive Modeling**

#### **Baseline Model Selection:**
Trained and compared 7 baseline models to establish performance benchmarks:
- **Logistic Regression:** Linear classifier with regularization
- **Decision Tree:** Non-parametric tree-based classifier
- **Random Forest:** Ensemble of decision trees with bagging
- **Gradient Boosting:** Sequential ensemble with gradient descent optimization
- **Support Vector Machine (SVM):** Kernel-based classifier
- **K-Nearest Neighbors (KNN):** Instance-based learning algorithm
- **Naive Bayes:** Probabilistic classifier based on Bayes theorem

#### **Model Training:**
- All models trained on preprocessed training set (320 samples)
- Default hyperparameters used for baseline comparison
- Consistent preprocessing pipeline applied to all models

#### **Cross-Validation:**
- 5-fold cross-validation performed for robust performance estimation
- Stratified k-fold to maintain class distribution across folds
- CV scores reported as mean ± standard deviation

### **6. Model Evaluation**

#### **Evaluation Metrics:**
- **Accuracy:** Overall correctness of predictions (98.75% test accuracy)
- **Precision:** Accuracy of positive predictions (96.77% - minimal false alarms)
- **Recall:** Coverage of positive cases (100% - zero missed CKD cases)
- **F1-Score:** Harmonic mean of precision and recall (98.36% - balanced performance)
- **ROC-AUC:** Area under ROC curve (100% - perfect class separation)
- **Cross-Validation F1-Score:** Mean F1 across 5 folds (98.79% ± 3.21%)

#### **Confusion Matrix Analysis:**
- True Positives (TP): Correctly predicted CKD cases
- True Negatives (TN): Correctly predicted non-CKD cases
- False Positives (FP): Incorrectly predicted CKD (false alarms)
- False Negatives (FN): Missed CKD cases (critical errors)

#### **Visualization:**
- Confusion matrix heatmap for best model
- ROC curve visualization showing discrimination ability
- Model comparison bar charts (accuracy, F1-Score, ROC-AUC)
- Feature importance visualization (where applicable)

#### **Model Selection:**
- Models ranked by F1-Score (primary metric for balanced performance)
- Logistic Regression selected as best model based on:
  - Highest F1-Score (98.36%)
  - Perfect recall (100% - critical for medical diagnosis)
  - High precision (96.77%)
  - Perfect ROC-AUC (100%)
  - Stable cross-validation performance (98.79% ± 3.21%)

---

## Results

**What did your research find?**

### **Key Findings:**

#### **Data Quality & Missing Values:**
- **Total Missing Values:** 1,002 missing values across 24 columns (out of 25 features)
- **Highest Missing Rates:** Red Blood Cells (38.0%), Red Blood Cell Count (32.75%), White Blood Cell Count (26.5%)
- **Missing Pattern:** Missing values are not randomly distributed - blood cell-related features have highest missing rates
- **Imputation Strategy:** KNN imputation (k=5) for numerical features and mode imputation for categorical features

#### **Top Predictors:**
1. **Serum Creatinine** - **408% difference** (CKD: 4.41 mg/dL vs Normal: 0.87 mg/dL) - Strongest indicator
2. **Blood Urea** - **121% difference** (CKD: 72.39 mg/dL vs Normal: 32.80 mg/dL) - Second best predictor
3. **Blood Glucose Random** - **63% difference** - Diabetes-related kidney damage indicator

#### **Clinical Patterns:**
- **Anemia Pattern:** CKD patients show consistent anemia (29.9% lower haemoglobin, 28.9% lower PCV, 26.7% lower RBCC)
- **Risk Factors:** CKD patients significantly more likely to have hypertension, diabetes mellitus, poor appetite, and abnormal urine findings
- **Electrolyte Imbalances:** Sodium and potassium show extreme skewness suggesting severe imbalances in subgroups

### **Model Performance:**

**Best Model:** Logistic Regression achieves highest F1-Score of 0.9836 (98.36%) among 7 baseline models tested.

**Test Accuracy:** Model achieves 98.75% accuracy on test set, correctly classifying 79 out of 80 patients.

**Precision:** 96.77% precision means when model predicts CKD, it's correct 96.77% of the time with minimal false alarms.

**Recall:** 100% recall means the model successfully identifies all CKD patients in the test set with zero missed cases.

**F1-Score:** 98.36% F1-Score demonstrates excellent balance between precision and recall for clinical diagnosis.

**ROC-AUC:** Perfect 100% ROC-AUC score indicates the model can perfectly distinguish between CKD and non-CKD patients across all decision thresholds.

**Cross-Validation:** CV F1-Score of 98.79% (+/- 3.21%) confirms consistent and stable model performance across different data splits.

**Model Comparison:** Logistic Regression outperforms Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, and Naive Bayes based on F1-Score ranking.

---

## Next Steps

**What suggestions do you have for next steps?**

- Hyperparameter Tuning (GridSearchCV)
- Feature Selection Refinement
- Ensemble Methods

---

## Contact and Further Information

For questions or additional details about this project, please refer to the main analysis notebook: [`ckd-prediction-eda.ipynb`](ckd-prediction-eda.ipynb)
