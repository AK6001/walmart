import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)


rf = pickle.load(open('reg1.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    format = request.args.get('format')
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = rf.predict(final_features)
    output = round(prediction[0],2)

    if(format == 'json'):
        return jsonify({'salary': output})

    return render_template('index.html', prediction_text='weekly sales should be $ {}'.format(output))
    
    
    
    
    if __name__ == "__main__":
      app.run(debug=True)  # auto-reload on code change
