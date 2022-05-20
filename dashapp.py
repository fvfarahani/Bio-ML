from dash import Dash, dcc, html, Input, Output
import os
import plotly.express as px
import pandas as pd
import numpy as np 
import joblib

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Enter your patients data to obtain a predicted result"),
    html.Div([
        html.H2('Age'),
        dcc.Input(id = 'age', value = 32, type = 'number'),
        html.H2('Sex'),
        dcc.Dropdown(options = [ {'label' : 'Male', 'value' : 1},
                                 {'label' : 'Female', 'value' : 0},],value = 1, id = 'sex'),
        html.H2('amount of albumin in patient blood'),
        dcc.Input(id = 'ALB', value = 38.5, type = 'number'),
        html.H2('amount of alkaline phosphatase in patient blood'),
        dcc.Input(id = 'ALP', value = 52.5, type = 'number'),
        html.H2('amount of alanine transaminase in patient blood'),
        dcc.Input(id = 'ALT', value = 7.7, type = 'number'),
        html.H2(' amount of aspartate aminotransferase in patient'),
        dcc.Input(id = 'AST', value = 22.1, type = 'number'),
        html.H2('amount of bilirubin in patient blood'),
        dcc.Input(id = 'BIL', value = 7.5, type = 'number'),
        html.H2('amount of cholinesterase in patient blood'),
        dcc.Input(id = 'CHE', value = 6.93, type = 'number'),
        html.H2('amount of cholesterol in patient blood'),
        dcc.Input(id = 'CHOL', value = 3.23, type = 'number'),
        html.H2('amount of creatine in patient blood'),
        dcc.Input(id = 'CREA', value = 106.0, type = 'number'),
        html.H2('amount of gamma-glutamyl transferase in patient blood'),
        dcc.Input(id = 'GGT', value = 12.1, type = 'number'),
        html.H2('amount of protien in patient blood'),
        dcc.Input(id = 'PROT', value = 69.0, type = 'number')

    ]),
    html.Br(),
    html.H1("Enter the prediction model"),
    dcc.Dropdown(options = [ {'label' : 'Random Forest', 'value' : 0},
                             {'label' : 'Gaussian Naive Bayes', 'value' : 1},
                             {'label' : 'Logistic Regression', 'value' : 2},
                             {'label' : 'K Nearest Neighbor', 'value' : 3},
                             {'label' : 'Support Vector Machine', 'value' : 4}],value = 0, id = 'model_choice'),
                             
    html.Br(),
    html.H1("Your prediction results: "),
    dcc.Graph(id = 'output-graph'),
    html.H1(id = 'WAIT FOR MODEL TRAINING')
])


@app.callback(
    Output('output-graph', 'figure'),
    # Output('output-graph', 'children'),
    Input('age', 'value'),
    Input('sex', 'value'),
    Input('ALP', 'value'),
    Input('ALB', 'value'),
    Input('ALT', 'value'),
    Input('AST', 'value'),
    Input('BIL', 'value'),
    Input('CHE', 'value'),
    Input('CHOL', 'value'),
    Input('CREA', 'value'),
    Input('GGT', 'value'),
    Input('PROT', 'value'),
    Input('model_choice', 'value')
)
def update_output_div(age, sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT, model_choice):

    dat = pd.DataFrame(np.array([[age, sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT]])
                        ,columns = [['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE','CHOL','CREA', 'GGT', 'PROT']])
    
    if model_choice == 0:
        model = joblib.load("Model/rf_model.joblib")
        fptr = 'Random Forest'
    elif model_choice == 1:
        model = joblib.load("Model/gnb_model.joblib")
        fptr = 'Gaussian Naive Bayes'
    elif model_choice == 2:
        model = joblib.load("Model/logReg_model.joblib")
        fptr = 'Logistic Regression'
    elif model_choice == 3:
        model = joblib.load("Model/knn_model.joblib")
        fptr = 'K Nearest Neighbor'
    elif model_choice == 4:
        model = joblib.load("Model/svc_model.joblib")
        fptr = 'Support Vector Machine'


    scalar = joblib.load("Model/scalar.joblib")
    x_new = scalar.transform(dat)
    # predict the heart failure prossibility of patient based on a chosen model
    prob = model.predict_proba(x_new)
    labels=['Predicted to have Hepatitis (Red color)','Predicted healthy(Green color)']
    data = {'results': [prob[0][1],prob[0][0]]}
    df = pd.DataFrame.from_dict(data)
    colors = ['firebrick', 'olive']
    fig = px.pie(df, values='results',names=labels, title='Predicted results based on %s model' % fptr, hole=.5,color=labels,
                color_discrete_map={'Predicted to have Hepatitis (Red color)':'darkorange','Predicted healthy(Green color)':'lightgreen'},width=800, height=500)

    return fig

if __name__ == '__main__':
    # app.run_server(host='jupyter.biostat.jhsph.edu', port = os.getuid() + 30)
    app.run_server(debug=True, host = '127.0.0.1')