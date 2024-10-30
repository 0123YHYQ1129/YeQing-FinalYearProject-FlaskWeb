from flask import Blueprint, render_template, request
import requests
import plotly.io as pio

train = Blueprint('train', __name__)

@train.route('/train', methods=['POST', 'GET'])
def train_function():
    if request.method == 'POST':
        json_data = {
            'Source_data': request.form.get('Source_data'),
            'Prediction_length': request.form.get('Prediction_length'),
            'Window_size': request.form.get('Window_size'),
            'Feature': request.form.get('Feature'),
            'Model': request.form.get('Model')
        }
        result = requests.post('http://localhost:5001/pred/data_process', json=json_data, timeout=600).json()
    else:
        json_data = {
            'Source_data': "ETTh1",
            'Prediction_length': 4,
            'Window_size': 64,
            'Feature': "MS",
            'Model': "GRU"
        }
        result = {key: None for key in ['test_r2', 'test_mae', 'test_rmse', 'loss_plot', 'train_r2', 'test_plot', 'train_plot']}

    result.update(json_data)
    return render_template('train/train.html', **result)
