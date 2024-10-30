from flask import request
import warnings
from flask_restx import Namespace, Resource, fields
from . import api_response
import json
import torch
from algorithm.data_process import create_dataloader
from algorithm.train import train
from algorithm.models.gru import GRU
from algorithm.test_and_inspect_fit import test, inspect_model_fit
from algorithm.predict import predict
import plotly.io as pio

pred = Namespace('pred', description='Process the time series into a data loader.')

# Model for the API parameters
train_model = pred.model('train_model', {
    'Source_data': fields.String(required=True, enum=['ETTh1', 'Shenzhen', 'Los_Angeles'], default='ETTh1', description='Choose your .csv file'),
    'Prediction_length': fields.Integer(required=True, default=4, description='The length of the prediction'),
    'Window_size': fields.Integer(required=True, default=64, description='The length of the time series used for input to the model'),
    'Feature':fields.String(required=True, enum=['MS', 'M'], default='MS', description='Multivariate prediction / Univariate prediction'),
    'Model':fields.String(required=True, enum=['GRU', 'TGCN'], default='GRU', description='Choose your model'),
})

predict_model = pred.model('predict_model', {
    'Source_data': fields.String(required=True, enum=['ETTh1', 'Shenzhen', 'Los_Angeles'], default='ETTh1', description='Choose your .csv file'),
    'Prediction_length': fields.Integer(required=True, default=4, description='The length of the prediction'),
    'Window_size': fields.Integer(required=True, default=64, description='The length of the time series used for input to the model'),
    'Feature':fields.String(required=True, enum=['MS', 'M'], default='MS', description='Multivariate prediction / Univariate prediction'),
    'Model':fields.String(required=True, enum=['GRU', 'TGCN'], default='GRU', description='Choose your model'),
})
@pred.route('/data_process')
class data_process(Resource):
    @pred.doc(responses=api_response)
    @pred.expect(train_model)
    def post(self):
        try:
            with warnings.catch_warnings(record=True) as w:
                if pred.payload['Source_data'] == 'ETTh1':
                    data_path =  'algorithm/dataset/ETTh1.csv'
                    target = 'OT'
                    input_size = 7
                elif pred.payload['Source_data'] == 'Shenzhen':
                    data_path = 'algorithm/dataset/ETTh1.csv'
                    target = '94161'
                    input_size = 40
                elif pred.payload['Source_data'] == 'Los_Angeles':
                    data_path = 'algorithm/dataset/los_speed.csv'
                    target = '767620'
                    input_size = 40
                pre_len = int(pred.payload['Prediction_length'])
                window_size = int(pred.payload['Window_size'])
                feature = pred.payload['Feature']
                device = torch.device("cpu")
                train_loader, test_loader, valid_loader, scaler, output_string = create_dataloader(data_path, pre_len, window_size, target, feature, device)
                if pred.payload['Model'] == 'GRU':
                    model = GRU(input_size=input_size, hidden_size=24, pre_len=pre_len, device=device).to(device)
                    loss_plot = train(model, feature, train_loader, scaler)
                    json_loss_plot = pio.to_json(loss_plot)
                    output_string = 'GRU is trained successfully!'
                elif pred.payload['Model'] == 'TGCN':
                    output_string = 'TGCN is not supported yet!'
                test_plot, test_r2, test_mae, test_rmse= test(model, feature, test_loader, scaler)
                train_plot, r2 = inspect_model_fit(model, train_loader, scaler)
                json_test_plot = pio.to_json(test_plot)
                json_train_plot = pio.to_json(train_plot)
                result_json = {'output':output_string,
                               'loss_plot':json_loss_plot,
                               'test_plot':json_test_plot,
                               'train_plot':json_train_plot,
                               'train_r2':float(r2),
                               'test_r2':float(test_r2),
                               'test_mae':float(test_mae),
                               'test_rmse':float(test_rmse),}  # Convert the string to a dictionary
                warning_list = []
                for warning in w:
                    # Add all warning messages to result_json
                    warning_list.append(str(warning.message))
                result_json["warning"] = warning_list
                return result_json, 201
        except ValueError as e:
            # Catch the ValueError message
            return {'test_r2':"ValueError",
                    'test_mae':"ValueError",
                    'test_rmse':"ValueError",
                "prediction_value": "Not available",
                "ValueError": "There is something wrong with your input!"
                + " Please check the range of values entered.",
            }, 400
        except KeyError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": str(e)}, 500
                    
@pred.route('/predict')
class draw_predict_cruve(Resource):
    @pred.doc(responses=api_response)
    @pred.expect(predict_model)
    def post(self):
        try:
            with warnings.catch_warnings(record=True) as w:
                if pred.payload['Source_data'] == 'ETTh1':
                    data_path =  'algorithm/dataset/ETTh1.csv'
                    target = 'OT'
                    input_size = 7
                elif pred.payload['Source_data'] == 'Shenzhen':
                    data_path = 'algorithm/dataset/ETTh1.csv'
                    target = '94161'
                    input_size = 40
                elif pred.payload['Source_data'] == 'Los_Angeles':
                    data_path = 'algorithm/dataset/los_speed.csv'
                    target = '767620'
                    input_size = 40
                pre_len = int(pred.payload['Prediction_length'])
                window_size = int(pred.payload['Window_size'])
                feature = pred.payload['Feature']
                device = torch.device("cpu")
                _, _, _, scaler, output_string = create_dataloader(data_path, pre_len, window_size, target, feature, device)
                if pred.payload['Model'] == 'GRU':
                    model = GRU(input_size=input_size, hidden_size=24, pre_len=pre_len, device=device).to(device)
                    pred_cruve = predict(model, data_path, window_size, device, scaler)
                    json_pred_cruve = pio.to_json(pred_cruve)
                    output_string = 'GRU is trained successfully!'
                elif pred.payload['Model'] == 'TGCN':
                    output_string = 'TGCN is not supported yet!'
                result_json = {'output':output_string,
                               'loss_plot':json_pred_cruve,
                }
                warning_list = []
                for warning in w:
                    # Add all warning messages to result_json
                    warning_list.append(str(warning.message))
                result_json["warning"] = warning_list
                return result_json, 201
        except ValueError as e:
            # Catch the ValueError message
            return {
                "prediction_value": "Not available",
                "ValueError": "There is something wrong with your input!"
                + " Please check the range of values entered.",
            }, 400
        except KeyError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": str(e)}, 500


