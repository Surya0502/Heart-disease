from flask import Flask,render_template,request
from flask_material import Material

# EDA PKg
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    return render_template("preview.html")

@app.route('/',methods=["POST"])
def analyze():
    if request.method == 'POST':
        age = request.form['age']
        education = request.form['education']
        currentSmoker = request.form['currentSmoker']
        cigsPerDay = request.form['cigsPerDay']
        BPMeds = request.form['BPMeds']
        prevalentStroke = request.form['prevalentStroke']
        prevalentHyp = request.form['prevalentHyp']
        diabetes = request.form['diabetes']
        totChol = request.form['totChol']
        sysBP= request.form['sysBP']
        diaBP  = request.form['diaBP']
        BMI = request.form['BMI']
        heartRate = request.form['heartRate']
        glucose = request.form['glucose']
        model_choice = request.form['model_choice']
        

		# Clean the data by convert from unicode to float 
        sample_data = [age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]
        clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
        if model_choice == 'logitmodel':
            logit_model = joblib.load('data/logit_model.pkl')
            result_prediction = logit_model.predict(ex1)
        elif model_choice == 'knnmodel':
            knn_model = joblib.load('data/knn_model.pkl')
            result_prediction = knn_model.predict(ex1)
        elif model_choice == 'svmmodel':
            svm_model = joblib.load('data/svm_model.pkl')
            result_prediction = svm_model.predict(ex1)
        elif model_choice == 'XGBoostmodel':
            xgb_model = joblib.load('data/xgb_model.pkl')
            result_prediction = xgb_model.predict(ex1)

    return render_template('index.html',age=age,education=education,currentSmoker=currentSmoker,cigsPerDay=cigsPerDay,BPMeds=BPMeds,prevalentStroke=prevalentStroke,prevalentHyp=prevalentHyp,diabetes=diabetes,totChol=totChol,sysBP=sysBP,diaBP=diaBP,BMI=BMI,heartRate=heartRate,glucose=glucose,clean_data=clean_data,result_prediction=result_prediction,model_selected=model_choice)


if __name__ == '__main__':
	app.run()

