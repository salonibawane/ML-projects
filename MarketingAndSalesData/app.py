from flask import Flask,render_template, url_for, request
from flask import jsonify
from utils import Sales
import pandas as pd
# Initialize Flask Application
app = Flask(__name__)



# Load the dataset to get dropdown options
dataset = pd.read_csv(r'data\Dummy Data HSS-Marketing and Sales Data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/influencer_options')
def region_options():
    return jsonify(list(dataset['Influencer'].unique()))

@app.route('/api/predict', methods=['POST'])
def predict():

    tv = request.form.get('tv')
    radio = request.form.get('radio')
    socialmedia = request.form.get('socialmedia')
    influencer = request.form.get('influencer')

    sales = Sales()
    prediction = sales.get_predicted_sales(tv,radio,socialmedia,influencer)
    # Make the prediction
  
    return jsonify({'prediction': prediction[0]})


if __name__ == "__main__":

    app.run(host='0.0.0.0',port=8080,debug=False)