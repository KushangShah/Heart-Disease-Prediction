<div align='left' style="margin-bottom: 40px;"><img height="30px" src="images/UM_Logo.png" alt="Unified Mentor Private Limited"></a><br>Unified Mentor Private Limited </div>
<br><br>
<!-- Image Logo -->
<div align="center">
    <img src="images/banner.png" alt="" width="400" height="235" style="border-radius: 18px;">
</div>
<br>

<!-- project and ML title -->
<h1 align='center' style="margin-bottom: 0px;">
<!-- <a href="https://git.io/typing-svg"> -->
  <img src="https://readme-typing-svg.herokuapp.com?font=Playfair+Display&weight=500&size=25&duration=5500&pause=1000&color=00FFFF&center=true&random=false&width=600&lines=Heart+Disease+Prediction" alt="Typing SVG" />
</a></h1>

<!-- ML name -->
<h4 align='center' style="margin-top: 0; margin-bottom: 10px;">
<!-- <a href="https://git.io/typing-svg"> -->
  <img src="https://readme-typing-svg.herokuapp.com?font=Playfair+Display&weight=100&size=15&duration=5500&pause=1000&color=0FFFFF&center=true&random=false&width=600&lines=Machine+Learning+Project" alt="Typing SVG" />
</a></h4>



---
<br>
<!-- Table of content -->
## Table of Contant

* <p style="font-size:14px;">About this Project.</p>
* <p style="font-size:14px;">Problem Statment.</p>
* <p style="font-size:14px;">What DataSet cointains?</p>
* <p style="font-size:14px;">Main library to be used.</p>
* <p style="font-size:14px;">Visualizations/Chart.</p>
* <p style="font-size:14px;">Conclusion.</p>
* <p style="font-size:14px;">Acknowledgment.</p>
<br>

---
<!-- About this Project -->
<br><br>
## About this Project:
<p style="font-size:15px;">
    This project focuses on predicting heart disease using a dataset containing various health indicators. We performed Exploratory Data Analysis (EDA) to understand the data, handled duplicates, and visualized key relationships. We then implemented and evaluated several machine learning models (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, and XGBoost), using cross-validation and hyperparameter tuning to optimize performance. The best-performing model, K-Nearest Neighbors (KNN), was saved for potential deployment. Key features influencing prediction were identified through the analysis.
    <br><br>
</p>

---
<br><br>
## Problem Statement:

Heart disease is a major global health concern. Early and accurate prediction is crucial for timely intervention and improved patient outcomes. This project aims to develop a machine learning model that can predict the likelihood of heart disease based on readily available patient health metrics, thereby assisting healthcare professionals in risk assessment and diagnosis.<br>

- Can we predict heart disease with high accuracy using readily available health data?<br>
- Which health indicators are the strongest predictors of heart disease in this dataset?<br>
- How do different machine learning models perform in predicting heart disease?<br>
- Can we identify patterns in patient data that indicate a higher risk of heart disease?<br>
- How can machine learning contribute to early diagnosis and prevention of heart disease?<br>
- Is there a significant difference in heart disease prevalence between genders in this dataset?<br>
<br><br>
---

<br><br>

<!-- What Data set Containes -->
<!-- columns and their descriptions -->
<!-- What Data set Containes -->
<!-- columns and their descriptions -->
<details open>
    <summary style="font-size: 20px; text-align: center;">
    What Data Set Contains?
</summary>
<br>
<p style="font-size:16px; test-align: left;">
1. age - Age of the patient.<br><br>
2. sex - Gender of the patient. Values: 1 = male, 0 = female.<br><br>
3. cp - Chest pain type. Values: 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic.<br><br>
4. resting_bp_s - Resting Blood Pressure (in mm Hg).<br><br>
5. cholesterol - Serum Cholesterol level (in mg/dl).<br><br>
6. fasting_blood_sugar - Fasting Blood Sugar > 120 mg/dl. Values: 1 = true, 0 = false.<br><br>
7. resting_ecg - Resting electrocardiographic results. Values: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy.<br><br>
8. max_heart_rate - Maximum heart rate achieved.<br><br>
9. exercise_angina - Exercise-induced angina. Values: 1 = yes, 0 = no.<br><br>
10. oldpeak - ST depression induced by exercise relative to rest.<br><br>
11. st_slope - Slope of the peak exercise ST segment. Values: 1 = Upward, 2 = Flat, 3 = Downward.<br><br>
12. target - Outcome variable (heart attack risk). Values: 1 = more chance of heart attack, 0 = less chance of heart attack.<br><br>
</p>
</details>
<br><br>

---
<br><br>
<!-- Library used in projects and their description -->
<details open> 
  <summary style="font-size: 20px; text-align:center;"> Main Libraries Used </summary> 
  <br> 
  <p style="font-size:16px;">
  🔢 <b>NumPy</b> – For efficient numerical computations, array operations, and mathematical functions.<br><br>
  📊 <b>Pandas</b> – To load, clean, and manipulate structured datasets for analysis and modeling.<br><br>
  📈 <b>Matplotlib & Seaborn</b> – For creating visualizations such as histograms, heatmaps, and correlation plots to explore data patterns and relationships.<br><br>
  🧪 <b>SciPy</b> – Used for performing statistical analysis, outlier detection, and hypothesis testing.<br><br>
  ⚙️ <b>scikit-learn (sklearn)</b> – The main machine learning library used for:
  <ul> 
    <li>📦 Data preprocessing (<code>StandardScaler</code>)</li> 
    <li>🧠 Model building (<code>LogisticRegression</code>, <code>KNeighborsClassifier</code>, <code>SVC</code>, <code>DecisionTreeClassifier</code>, <code>RandomForestClassifier</code>)</li> 
    <li>🎯 Model evaluation (<code>accuracy_score</code>, <code>precision_score</code>, <code>recall_score</code>, <code>f1_score</code>, <code>roc_auc_score</code>, <code>confusion_matrix</code>, <code>ConfusionMatrixDisplay</code>)</li> 
    <li>🧩 Model tuning and validation (<code>train_test_split</code>, <code>cross_val_score</code>, <code>GridSearchCV</code>, <code>RandomizedSearchCV</code>)</li>
  </ul>
  ⚡ <b>XGBoost</b> – A powerful gradient boosting algorithm used for improving prediction performance and handling complex data patterns.<br>
  </p> 
</details>
<br><br>

---
<br><br>
<details open>
<summary style="font-size: 20px; text-align:center;"> Visualizations/Chart. </summary>
<p align="center">
    <img src="images/Age.png" alt="Insights Visualization 1" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/cp_target.png" alt="Insights Visualization 2" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/max_hr.png" alt="Insights Visualization 3" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/rf_feature_importance.png" alt="Insights Visualization 4" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/logreg.png" alt="Insights Visualization 5" width="400" height="160" style="border-radius: 20px; margin: 10px;">
    <img src="images/ht_logreg.png" alt="Insights Visualization 6" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/KNN.png" alt="Insights Visualization 7" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/ht_knn.png" alt="Insights Visualization 8" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/SVM.png" alt="Insights Visualization 9" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/ht_SVM.png" alt="Insights Visualization 10" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/dt.png" alt="Insights Visualization 11" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/ht_dt.png" alt="Insights Visualization 12" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/rf.png" alt="Insights Visualization 13" width="400" height="160" style="border-radius: 20px; margin: 10px;">
    <img src="images/dt_rf.png" alt="Insights Visualization 14" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/xgb.png" alt="Insights Visualization 15" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/dt_xgb.png" alt="Insights Visualization 16" width="400" style="border-radius: 20px; margin: 10px;">
    <img src="images/confusion_matric.png" alt="Insights Visualization 17" width="400" style="border-radius: 20px; margin: 10px;">
</p>

---
<br><br>
## Conclusion:
<p style="font-size:16px;">
This project aimed to predict heart disease based on various health indicators. We performed extensive Exploratory Data Analysis (EDA) to understand the dataset, identifying key features and their distributions, as well as checking for missing and duplicate values.
Several machine learning models were implemented and evaluated, including Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, and XGBoost. Cross-validation and hyperparameter tuning were applied to improve model performance.
Based on the evaluation metrics, particularly accuracy, precision, recall, F1 score, and ROC-AUC, the K-Nearest Neighbors (KNN) model with hyperparameter tuning showed the best performance on the test set, achieving an accuracy of approximately 95%. This suggests that KNN is the most suitable model for this dataset in predicting heart disease.
The analysis also highlighted important features like ST slope, chest pain type, and max heart rate as significant predictors of heart disease.
In conclusion, this project successfully built and evaluated several models for heart disease prediction, with the tuned KNN model demonstrating promising results. The saved model can be used for future deployment to predict heart disease on new, unseen data.
</p>
<br><br>

---
<br><br>
### Acknowledgments:
<p style="font-size: 11px;">
This project is dedicated to applying machine learning techniques to understand and predict heart disease, contributing to early diagnosis and better healthcare outcomes. Sincere thanks to Unified Mentor Private Limited for providing the opportunity and platform to carry out this work. Appreciation is also extended to the open-source community for developing the powerful tools and libraries that made this project possible.
</p>
<p align="right" > Created with 🧠 by <a href="https://github.com/KushangShah">Kushang Shah</a></p>
<p align="right"> <img src="https://komarev.com/ghpvc/?username=kushang&label=Profile%20views&color=0e75b6&style=flat" alt="kushang" /> </p>