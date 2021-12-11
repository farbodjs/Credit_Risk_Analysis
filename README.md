# Credit_Risk_Analysis
## Overview of project
In this project, our goal is to use machine learning techniques to analyzed our data (LoanStats_2019Q1) which contains 115676 rows of applicants and 86 columns of factors.Our goal is to perform analysis and figure out how all these factors in our loan_stats csv help predict whether a applicant is low or high risk for lending purposes. We created several machine learning models and then evaluated and train the models to calculate accuracy score and precision. In this specific project we are using imbalanced-learn and scikit-learn libraries to build models and evalute them using various resampling methods. In the first couple of models we used  oversampling method using randomoversampler library and smote algorithms and undersample the data with the clustercentroid algorithms. In the remaining models we used a combination approach of these models. We analyzed performance and efficiency of each model by creating confusion matrix and running classification report.
## Results of Resampling techniques ( please refer to credit_risk_resampling.ipynb)
### Results for Naive Random Oversampling using RandomOverSampler library 
Our balanced accuracy test it 64%, the precision for the high_risk has a very low positivity at 1% and the recall is 71%. However, we achieved a precision of 100% for Low_risk which means our prediction of low_risk had no faults. Overall, this model is not satisfactory since it fails to predict high_risk candidates
![image](https://user-images.githubusercontent.com/86033316/145658196-3938c12e-c195-4e56-bcb7-6c8c46dbf422.png)

### Results for SMOTE Oversampling using SMOTE Library
The accuracy score is 65.9%, the precision for the high_risk loans has a low precision at 1% and recall is 68% overall for average of low_risk and high_risk.This Model also doesn not produce satisfactory results.
![image](https://user-images.githubusercontent.com/86033316/145658352-75ed37cd-ca68-4001-949e-4434dae81643.png)
### Results for Undersampling using ClusterCentroids Library
The accuracy score is 65.9%, the precision for the high_risk has a very low prescicion at 1% and recall is 69% for high_risk and 40% for low_risk which is again not satisfactory.
![image](https://user-images.githubusercontent.com/86033316/145658599-0b0a3e6e-bdba-4216-ad56-cce73dc991ae.png)
### Results for combination of (over and under)sampling using SMOTEENN Library
The accuracy score for this model is even lower than the previous ones with 54% accuracy. The precision for high_risk is 1 % and low_risk is 99%. Recall for high_risk is 72% and for low_risk is 57%.

![image](https://user-images.githubusercontent.com/86033316/145658813-ec474a6a-42c6-44ff-92c1-22a7ef76b2a4.png)

## Results of ensembling techniques ( please refer to credit_risk_ensemble.ipynb)

### Results for ensembling using BalancedRandomForestClassifier Library
The accuracy score is slightly improved to 75% accuracy. Precision for high_risk is still low as 3% but is improved from resampling techniques. Recall factor for high_risk is 60% and for low_risk is 89%
![image](https://user-images.githubusercontent.com/86033316/145658995-58152b23-cd1a-4bc3-9e71-bd033382e7f2.png)


Subsequently, we listed the features sorted in descending order by feature importance. Please see below snippet for factors with least importance based on BalancedRandomForestClassifier. We can try elliminating some of these unimportant factors to improve precision of our model
![image](https://user-images.githubusercontent.com/86033316/145659189-10eb0d36-a03c-45ec-9904-17456fd121eb.png)

### Results for ensembling using EasyEnsembleClassifier Library
So far, this is the most efficient model with an accuracy of 94% and precision of 10% for high_risk and 100% for low_risk. the recal factor for high_risk is 92% and 95% for low_risk. These results are satisfactory and we can see that precision for high_risk is ten times greater than resampling models.
![image](https://user-images.githubusercontent.com/86033316/145659237-19533349-4710-4780-88de-29520ebefc74.png)


### Summary of findings

In the first four models of resampling, we undersampled, oversampled and utilized a combination of both methods to try and determine which model is more accurate at predicting the risk of loan application. As stated in above paragraphs showing each technique's results we concluded neighter of these methods can produce accurate and satisfactory results. We then tried using ensemble classifiers to try and predict which which loans are high or low risk. We concluded that EasyEnsembleClassifier technique produces more robust results.Typically, it is desired that a model shows good balance of recall and precision which is why we recommend the ensemble classifiers over the other five models. It appears that the Easy Ensemble had the best balance of all the models because of it's high accuracy score and good balance of precision and recall scores.
