## Background
The project plan establish a prediction model based on an [open database](https://www.kaggle.com/fedesoriano/hepatitis-c-dataset) which contains  615 observations and 14 attributes of blood donors and Hepatitis C patients. Machine learning algorithms such as logistic regression, random forests, support vector machine are possible choices for model fitting. Here our group aims to build prediction web app based on those data which allow Hepatitis prediction for new variants of a patient. 

## Setup instructions
1) git clone the repo
2) do ```python pip install ``` to install the repo below:  

   **Data Address:**  ```pandas```  ```numpy```  ```joblib``` ```sklearn```  

   **Draw Figures:**  ```matplotlib``` ```plotly```

   ***Web Interface:** ```dash``` ```os```
3) do ```python dashapp.py ``` to run the web app locally. 


## How our data look like ?

***The Features***

**Category:** The target feature. values: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'

**Age:** age of the patient in years

**Sex:** sex of the patient ('f'=female, 'm'=male)

**ALB:** amount of albumin in patient's blood

**ALP:** amount of alkaline phosphatase in patient's blood

**ALT:** amount of alanine transaminase in patient's blood

**AST:** amount of aspartate aminotransferase in patient's blood

**BIL:** amount of bilirubin in patient's blood

**CHE:** amount of cholinesterase in patient's blood

**CHOL:** amount of cholesterol in patient's blood

**CREA:** amount of creatine in patient's blood

**GGT:** amount of gamma-glutamyl transferase in patient's blood

**PROT:** amount of protien in patient's blood

<img width="900" height="400" src=https://github.com/fvfarahani/LeLiFa/blob/08f7dde69a22d2b2c460367c5426c9591cc68f0c/Figure/Features.png>

***The Labels***

Here we shows the labels of our dataset. 

<img width="500" height="150" src=https://github.com/fvfarahani/LeLiFa/blob/598561f1199ab07989c6fc6cfacc8018143db78e/Figure/Labels.png>


## How does our model perform?

***Algorithms***

**K-nearest Neighbors (KNN):** The KNN algorithm is a data classification approach that estimates the likelihood that a data point will belong to one of the available groups based on the data points that are closets to it. We set the number of neighbors to 1.

**Logistic Regression (LR):** Logistic regression is used for modeling the probability of a discrete outcome given an input variable. We used L2 regularization, also known as ridge regression, which adds the “squared magnitude” of the coefficient to the loss function as the penalty term.

**Gaussian Naive Bayes (GNB):** A Gaussian Naive Bayes algorithm is a special type of NB algorithm that is employed when the features have continuous values. Features are assumed to have a gaussian distribution. The var smoothing option was set to 1e-09, which is the percentage of the largest variance of all features that is applied to variances for computation stability.

**Random Forest (RF):** A random forest is a meta estimator that fits multiple decision tree classifiers on different sub-samples of the dataset and employs averaging to increase the predictive accuracy and control over-fitting. We used 10 trees (estimators) in the forest.

**Support Vector Machine (SVM):** We performed grid search to optimize hyperparameters, and selected "rbf" as kernel and set the number of C to 1.

***Model Performance***

<img width="500" height="250" src=https://github.com/fvfarahani/LeLiFa/blob/06e9dac7410b413d1cc51211016eb56eb5d392eb/Figure/performance.png>

***ROC and PR Curve***

<img width="600" height="300" src=https://github.com/fvfarahani/LeLiFa/blob/06e9dac7410b413d1cc51211016eb56eb5d392eb/Figure/roc.png>

## How can our dash web app help you predict the Hepatitis?

***Input a new patient information***

<img width="300" height="400" src=https://github.com/fvfarahani/LeLiFa/blob/600a29a8c016bfecdf091391f24aa5298bf0d808/Figure/dash1.png>

***Choose a model***

<img width="400" height="120" src=https://github.com/fvfarahani/LeLiFa/blob/600a29a8c016bfecdf091391f24aa5298bf0d808/Figure/dash2.png>

***Obtain the prediction result***

<img width="500" height="300" src=https://github.com/fvfarahani/LeLiFa/blob/600a29a8c016bfecdf091391f24aa5298bf0d808/Figure/dash3.png>


## Contributors
This project exists thanks to all the people who contribute. 
 
Lelin Zhong: lzhong6@jhu.edu

Lingzhu Shen: lshen26@jhu.edu, 

Farzad Farahani: ffaraha2@jhu.edu