# Machine Learning Project Summary - Supervised Learning

## Project Objectives
In this project, I apply supervised learning techniques to build a machine-learning model that can predict whether a patient has diabetes based on specific diagnostic measurements. The ultimate goal of the project is to gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked.

## Project Process
The dataset used in this project is the "Diabetes" dataset obtained from [Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset). See the data folder for the CSV file. The project involves three main parts: exploratory data analysis, preprocessing and feature engineering, and training a machine learning model. 

Overall, the process involved:
* Importing the dataset.
* Analyzing and visualizing the relationships between different variables to gain insights.
* Addressing missing values and outliers.
* Standardizing feature scales to ensure fairness in model training.
* Carrying out essential feature engineering when necessary.

At the end of the preliminary data preparation steps, the project proceeded to build machine learning models with the goal of training models capable of predicting the presence or absence of diabetes. In this step, Random Forest and Logistic Regression models were employed using the Train/Test Split and K_Fold Cross-Validation methods to evaluate the following metrics, such as accuracy, precision, recall, F1-score, and ROC_AUC to compare the model's performance (See the Supervised Learning - Project. ipynb in the notebook folder).

## Project Results

### Exploratory Data Analysis and Feature Engineering 
* During the EDA step, there were all the values. However, I observed possible irregularities in the data entry process or the possibility that patient records were not measured and were instead recorded as 0 rather than NULL. Rather than discarding these records with zero values, median values of the respective features were imputed, given the relatively small size of the dataset.

* Also, the boxplot of each predictor variable shows some outliers, with the feature 'Insulin' having significantly higher values than the rest of the data. Given the small size of the dataset, the capping method was employed to handle outliers for simplicity rather than removing these values.

* Feature scaling was carried out to address the variations in unit representation and magnitudes across different variables. This ensured fair and practical contributions from all features during the learning process.


### Machine Learning Model Training
 * Evaluated two distinct machine learning models, Logistic Regression and random forest classifiers, and compared their performance using evaluation metrics, including accuracy, precision, recall, F1-score, and ROC-AUC.
 * The performance of two machine learning models, Logistic Regression and Random Forest Classifier, was compared. Logistic Regression showed superior accuracy and precision, while Random Forest demonstrated proficiency in classification tasks with higher Recall and F1 score values.
 * Comparing the evaluation methods (Train/Test Split and the K-Fold Cross Validation), the performance metrics for both models have decreased when using K-Fold Cross Validation. This is not uncommon and can be because K-Fold Cross Validation provides a more robust estimate of model performance by averaging the metrics across multiple subsets of the data.
 * When considering selected features, Logistic Regression outperformed Random Forest regarding accuracy, precision, recall, F1 score, and ROC-AUC.
 * Overall, Logistic Regression emerged as the preferred model for this binary classification task. 

## Challenges 
Several challenges were encountered during the project, including time constraints, decision-making regarding outlier and incorrect data handling, and selecting the most suitable machine learning models for analysis. 

## Future Considerations

* Explore different feature selection approaches, including top correlated features and utilize all available features to assess their impact on model performance.
* Conduct hyperparameter tuning and feature importance for the random forest ensemble model to enhance its performance and estimate the importance of features.
* Also, examine other machine learning models, such as SVM and GradientBoosting, for further analysis.
