import pickle
from flask import Flask , request , app , jsonify , url_for , render_template
import numpy as np 
import pandas as pd 


# starting point to run from 
app = Flask(__name__ , template_folder='/Users/sly/Desktop/Boston Housing Price/simple_end_to_end_bostenhousepricing/templates')
# Loading pretrained model 
reg_model = pickle.load(open('/Users/sly/Desktop/Boston Housing Price/simple_end_to_end_bostenhousepricing/reg_model.pkl' , 'rb'))
# loading in scaler 
scaler = pickle.load(open('/Users/sly/Desktop/Boston Housing Price/simple_end_to_end_bostenhousepricing/Notebooks/scaler.pkl' , 'rb'))
# create app.route 

@app.route('/')
# open home html 
def home():
    return render_template('home.html')

# create predict api
@app.route('/predict.api' , methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = reg_model.predict(new_data)
    return jsonify(output[0])


@app.route('/predict' , methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_inputs = scaler.transform(np.array(data).reshape(1,-1))
    output = reg_model.predict(final_inputs)
    return render_template('home.html' , prediction_text = 'The House price prediction is {}'.format(output[0]))

if __name__ == '__main__':
    app.run(debug=True)

