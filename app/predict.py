from flask import Blueprint, render_template, request
import requests
import plotly.io as pio

predict = Blueprint('predict', __name__)

@predict.route('/predict', methods=['POST', 'GET'])
def predict_function():
    if request.method == 'POST':
        json_data = {
            'Source_data': request.form.get('Source_data'),
            'Prediction_length': request.form.get('Prediction_length'),
            'Window_size': request.form.get('Window_size'),
            'Feature': request.form.get('Feature'),
            'Model': request.form.get('Model')
        }
        result = requests.post('http://localhost:5001/pred/predict', json=json_data, timeout=600).json()
    else:
        json_data = {
            'Source_data': "ETTh1",
            'Prediction_length': 4,
            'Window_size': 64,
            'Feature': "MS",
            'Model': "GRU"
        }
        result = {key: None for key in ['loss_plot']}

    result.update(json_data)
    return render_template('predict/predict.html', **result)
