from flask import Flask,render_template, url_for, request
from flask import jsonify
from utils import Sales
# Initialize Flask Application
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

# Define Route
@app.route('/api/predict',methods=['POST'])
def predict_sales():
    print("*"*50)

    tv = request.form.get('tv')
    radio = request.form.get('radio')
    newspaper = request.form.get('newspaper')

    sales = Sales()

    result = sales.get_predicted_sales(tv, radio, newspaper)

    return jsonify({'prediction':result[0]})


if __name__ == "__main__":
    app.run(debug=False,port=5002)