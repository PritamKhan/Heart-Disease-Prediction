from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

@app.get("/")
def start():
    return render_template('index.html', prediction="Not calculated yet")

@app.post('/')
def predict():
    model = joblib.load('ML_Projects/model.joblib')

    Age = request.form.get('Age')
    Sex = request.form.get('Sex') 
    CP = request.form.get('CP')
    Trestbps = request.form.get('Trestbps')
    Chol = request.form.get('Chol') 
    FBS = request.form.get('FBS') 
    Restecg = request.form.get('Restecg')
    Thalach = request.form.get('Thalach')
    Exange = request.form.get('Exange') 
    Oldpeak = request.form.get('Oldpeak') 
    Slope = request.form.get('Slope')
    CA = request.form.get('CA')
    Thal = request.form.get('Thal')
    input = [Age, Sex, CP, Trestbps, Chol, FBS, Restecg, Thalach, Exange, Oldpeak, Slope, CA, Thal ]   
    prediction = model.predict(np.array(input).reshape(1, -1))
    result = 'You Are Healthy' if prediction == 0 else 'Consult Your Doctor'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
