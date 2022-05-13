import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import joblib
from tkinter import * 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve




def plot_confusion_matrix(y_test, y_pred,modelname):
    '''
    This function read the true outcome and the predicted outcome,
    and draw a confusion matrix of specific model

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred: *array*
            The predicted outcome
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    # calculate the confusion matrix based on different model 
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(conf_matrix, cmap='GnBu', alpha=0.75)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large') 
    plt.xlabel('Predictions', fontsize=10)
    plt.ylabel('Actuals', fontsize=10)
    plt.title('Confusion Matrix: %s '% modelname, fontsize=12)
    fig.savefig('Figure/%s_confusion_matrix.png' % modelname)


def ROC_curve(y_test, y_pred_proba,modelname):
    '''
    This function read the true outcome and the probablity of the positive predicted outcomes,
    and draw a ROC curve of specific model

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred_proba: *array*
            The probablity of the positive predicted outcomes
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    # calculate the false positive rate and true positive rate of model
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
    # calculate the auc value of model
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, 'b', label = '%s (AUC = %0.2f)' % (modelname,roc_auc))
    #plot the ROC curve and calculate AUC
    plt.plot([0, 1], [0, 1],'r--', label='No Skill Classifier')
    #plot the 'No Skill Classifier' curve
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
    #Plot the 'perfect performance' curve
    plt.legend(loc = 'lower right')
    plt.title('ROC Curve: %s '% modelname)
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlabel('False Positive Rate (1-specificity)')
    fig.savefig("Figure/%s_ROC_curve.png" % modelname)
    

def PR_curve(y_test, y_pred_proba,modelname):
    '''
    This function read the true outcome and the probablity of the positive predicted outcomes,
    and draw a ROC curve of specific model

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred_proba: *array*
            The probablity of the positive predicted outcomes
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    precision, recall, _= precision_recall_curve(y_test,y_pred_proba[:,1])
    #calculate precision and recall
    no_skill = len(y_test[y_test==1]) / len(y_test)
    fig = plt.figure()
    plt.plot(recall, precision, marker='.', label='%s ' % modelname)
    #plot PR curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill Classifier')
    #plot 'No Skill Classifier' curve
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve: %s '% modelname)
    plt.legend()
    fig.savefig("Figure/%s_PR_curve.png" % modelname)


def Scores(y_test,y_pred,y_pred_proba,modelname):
    '''
    This function read the true outcome, the predicted outcome 
    and the probablity of the positive predicted outcomes. 
    Then, it give back the precison, recall, F1 score and the ROC-AUC score of the model.

    **Parameters**
        y_test: *array*
            The real outcome
        y_pred: *array*
            The predicted outcomes
        y_pred_proba: *array*
            The probablity of the positive predicted outcomes
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    print('----------------------------')
    print('This is %s'% modelname)
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    #Calculate the scores by test set and probability estimates provided by the predict_pred
    print('ROC-AUC Score: %.3f' % roc_auc_score(y_test, y_pred_proba[:,1]))
    #Calculate the ROC-AUC score by test set result probability estimates provided by the predict_proba
    return None
    

def plot_learning_curve(dataset, estimator,modelname):
    '''
    This function read the dataset and give back its learning curve based on specific model.

    **Parameters**
        dataset: *dataframe*
            The dataframe used for model fitting
        estimator: *array*
            The fitting model
        modelname: *str*
            The name of model

    **Return**
        None
    '''
    x=dataset.drop(columns='HeartDisease')
    #Dataset except target
    y=dataset['HeartDisease']
    #Dataset of target
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y)
    #Number of samples in training set, score if training set, score of test set
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #Mean of test set scores
    test_scores_std = np.std(test_scores, axis=1)
    #Standard deviation of test scores
    fig = plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title('Learning Curve: %s '% modelname, fontsize='small')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b") 
    #Plot the learning curve with upper and lower limits of training score
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
    #Plot the learning curve with upper and lower limits of test score
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    #Plot learning curve
    plt.legend(loc="best")
    fig.savefig("Figure/%s_Learning_curve.png" % modelname)



def RandomForest(dataset):
    '''
    This function read a dataset and train it on random forest model.

    **Parameters**
        dataset: *dataframe*
            The addressed dataframe

    **Return**
        None
    '''
    X = dataset.drop(['HeartDisease'], axis=1)
    #Dataset except target
    Y = dataset['HeartDisease']
    #Dataset of target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
    #Split data to training set and test set
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    #Feature scaling
    rf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    rf.fit(x_train,y_train)
    #Fit the model to training set
    y_pred=rf.predict(x_test)
    #Predict the result of test set
    y_pred_proba=rf.predict_proba(x_test)
    #Predict the result of test set to plot ROC and calculate AUC
    joblib.dump(rf, "Model/rf_model.joblib" ,compress=1)
    plot_confusion_matrix(y_test,y_pred,'RandomForest')
    ROC_curve(y_test, y_pred_proba,'RandomForest')
    PR_curve(y_test, y_pred_proba,'RandomForest')
    plot_learning_curve(dataset,rf,'RandomForest')
    Scores(y_test,y_pred,y_pred_proba,'RandomForest')
    return y_test, y_pred, y_pred_proba



if __name__ == '__main__':

    ## Read & Address the data and do feature exploration ##
    #######################################################
    df = pd.read_csv('HepatitisCdata.csv')


    ## Use machine learning model to make classification ##
    ########################################################
    RandomForest(df)
