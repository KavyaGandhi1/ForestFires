from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler




application = Flask(__name__)
app = application

##lasso model
model = pickle.load(open('lasso.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home",methods=['GET','POST'])
def predict():
    prediction = None
    error = None

    if request.method == "POST":

        try:
            # Collect numerical inputs
            TEMPERATURE = float(request.form["Temperature"])
            RH = float(request.form["RH"])
            WS = float(request.form["Ws"])
            RAIN = float(request.form["Rain"])
            FFMC = float(request.form["FFMC"])
            DMC = float(request.form["DMC"])
            
            ISI = float(request.form["ISI"])
            

            # Collect categorical inputs
            CLASSES = int(request.form["Classes"])
            REGION = int(request.form["Region"])

            # Maintain the exact feature order used during training
            features = np.array([[
                TEMPERATURE, RH, WS, RAIN, FFMC, DMC, ISI, CLASSES, REGION
            ]])

            # Predict FWI
            result = model.predict(scaler.transform(features))[0]
        
            prediction = round(float(result), 2)

        except (ValueError, KeyError) as e:
            error = "Please enter valid values in all fields."

    return render_template(
        "home.html",
        prediction=prediction,
        error=error
    )
    



if __name__ == "__main__":
    app.run(debug=True)