from flask import Flask, render_template, request, jsonify
import pickle  # Import for model loading
from flask_cors import CORS, cross_origin

app = Flask(__name__, template_folder='template')
CORS(app)

model = pickle.load(open('model.pkl', 'rb'))

features = [
    'Current Assets', 'Cost of Goods Sold', 'EBITDA', 
    'Inventory','Net Income', 'Total Receivables', 
    'Market Value', 'Net Sales','Total Assets', 'Total Long-term Debt', 
    'Gross Profit','Total Current Liabilities', 'Retained Earnings', 
    'Total Revenue','Total Liabilities', 'Total Operating Expenses'
]

@app.route("/")
def index():
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  
    data_features = [data.get(name) for name in features]
    prediction = model.predict([data_features])[0]
    if prediction == 1:
        prediction = "Company is Alive"
    else:
        prediction = "Company has Failed"

    return jsonify({"Prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)