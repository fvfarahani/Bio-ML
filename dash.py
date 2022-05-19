
from dash import Dash, dcc, html, Input, Output
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import joblib

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Enter your data to see the results"),
    html.Div([
        html.H2('Age'),
        dcc.Input(id = 'age', value = 32, type = 'number'),
        html.H2('Sex'),
        dcc.Dropdown(options = [ {'label' : 'Male', 'value' : 1},
                                 {'label' : 'Female', 'value' : 0},],value = 1, id = 'sex'),
        html.H2('amount of albumin in patient blood'),
        dcc.Input(id = 'ALB', value = 60, type = 'number'),
        html.H2('amount of alkaline phosphatase in patient blood'),
        dcc.Input(id = 'ALP', value = 1.6, type = 'number'),
        html.H2('amount of alanine transaminase in patient blood'),
        dcc.Input(id = 'ALT', value = 60, type = 'number'),
        html.H2(' amount of aspartate aminotransferase in patient'),
        dcc.Input(id = 'AST', value = 1.6, type = 'number'),
        html.H2('amount of bilirubin in patient blood'),
        dcc.Input(id = 'BIL', value = 60, type = 'number'),
        html.H2('amount of cholinesterase in patient blood'),
        dcc.Input(id = 'CHE', value = 1.6, type = 'number'),
        html.H2('amount of cholesterol in patient blood'),
        dcc.Input(id = 'CHOL', value = 60, type = 'number'),
        html.H2('amount of creatine in patient blood'),
        dcc.Input(id = 'CREA', value = 1.6, type = 'number'),
        html.H2('amount of gamma-glutamyl transferase in patient blood'),
        dcc.Input(id = 'GGT', value = 60, type = 'number'),
        html.H2('amount of protien in patient blood'),
        dcc.Input(id = 'PROT', value = 1.6, type = 'number')

    ]),
    html.H1("Enter the prediction model"),
    dcc.Dropdown(options = [ {'label' : 'Random Forest', 'value' : 0},
                                 {'label' : 'Female', 'value' : 1},],value = 0, id = 'model_choice'),
    html.Br(),
    html.H1("Your prediction results: "),
    dcc.Graph(id = 'output-graph')
])


@app.callback(
    Output('output-graph', 'figure'),
    Input('age', 'value'),
    Input('sex', 'value'),
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
def update_output_div(age, sex, ALB, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT,model_choice):
    dat = pd.DataFrame(np.array([[32,1,38.5,52.5,7.7,22.1,7.5,6.93,3.23,106.0,12.1,69.0]])
                        ,columns = [['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE','CHOL', 'CREA', 'GGT', 'PROT']])
    x_new = dat
    model = joblib.load("Model/rf_model.joblib")
    # predict the heart failure prossibility of patient based on a chosen model
    prob = model.predict_proba(x_new)
    if model_choice == 0:
        modelname = "RandomFrest"

    name = prob_draw(prob[0][1],prob[0][0],modelname)

    def prob_draw(positive,negative,fptr):
        '''
        This function draw the pie chart of prediction results 

        **Parameters**
            positive: *int*
                The predicted probability of heart failure
            negative: *int*
                The predicted probability of being healthy
            fptr: *str*
                The name of the model
            
        **Return**
            The name of the pie chart figure.
        '''
        labels=['Predicted to have heart failure(Red color)','Predicted healthy(Green color)']
        X=[positive,negative]  
        colors = ['firebrick', 'olive']
        fig = plt.figure(figsize=(8, 4))
        plt.pie(X,labels=labels,autopct='%1.2f%%',colors = colors) 
        plt.title("Predicted results: %s" % fptr)
        fig.savefig("Figure/%s_PieChart.png" % fptr)
        return "Figure/%s_PieChart.png" % fptr

    return name

if __name__ == '__main__':
    app.run_server(host='jupyter.biostat.jhsph.edu', port = os.getuid() + 30)
    # app.run_server(debug=True, host = '127.0.0.1')