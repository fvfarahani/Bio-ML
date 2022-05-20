## Background
The project plan establish a prediction model based on an [open database](https://www.kaggle.com/fedesoriano/heart-failure-prediction) which contains  615 observations and 14 attributes of blood donors and Hepatitis C patients. Machine learning algorithms such as logistic regression, random forests, support vector machine are possible choices for model fitting. Here our group aims to build prediction web app based on those data which allow Hepatitis prediction for new variants of a patient. 


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

***Model Performance***

<img width="500" height="250" src=https://github.com/fvfarahani/LeLiFa/blob/06e9dac7410b413d1cc51211016eb56eb5d392eb/Figure/performance.png>

***ROC and PR Curve***

<img width="600" height="300" src=https://github.com/fvfarahani/LeLiFa/blob/06e9dac7410b413d1cc51211016eb56eb5d392eb/Figure/roc.png>

## How can our dash web app help you predict the heart failure?

1) Input a new patient information 
<img width="600" height="800" src=https://github.com/fvfarahani/LeLiFa/blob/600a29a8c016bfecdf091391f24aa5298bf0d808/Figure/dash1.png>

2) Choose a model 

<img width="500" height="170" src=https://github.com/fvfarahani/LeLiFa/blob/600a29a8c016bfecdf091391f24aa5298bf0d808/Figure/dash2.png>

3) Obtain the prediction result
<img width="500" height="300" src=https://github.com/fvfarahani/LeLiFa/blob/600a29a8c016bfecdf091391f24aa5298bf0d808/Figure/dash3.png>


## Contributors
This project exists thanks to all the people who contribute. 
 
Lelin Zhong: lzhong6@jhu.edu

Lingzhu Shen: lshen26@jhu.edu, 

Farzad Farahani: ffaraha2@jhu.edu