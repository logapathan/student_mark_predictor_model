from flask import Flask, render_template, request
import os   
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app=application

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        final_new_data=data.get_data_as_dataframe()
        

        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
        return render_template('home.html', results=pred[0])

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
