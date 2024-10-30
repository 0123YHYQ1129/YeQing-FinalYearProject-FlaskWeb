import torch
from sklearn.metrics import r2_score
import numpy as np
import plotly.graph_objects as go

def test(model, feature, test_loader, scaler):
    mae_list = []
    rmse_list = []
    model = model
    model.load_state_dict(torch.load( "model.pth"))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        if feature == 'MS':
            pred = pred[:, :, -1]
            label = label[:, :, -1]
        mae = calculate_mae(pred.detach().cpu().numpy(),
                            np.array(label.detach().cpu()))
        rmse = calculate_rmse(pred.detach().cpu().numpy(),
                            np.array(label.detach().cpu()))
        if feature == 'M':
            pred = pred[:, :, -1]
            label = label[:, :, -1]
        pred = pred[:, 0]
        label = label[:, 0]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        mae_list.append(mae)
        rmse_list.append(rmse)
        for i in range(len(pred)):
            results.append(pred[i])
            labels.append(label[i])

    print("Testset Mean Absolute Error(测试集平均绝对误差):", np.mean(mae_list))
    print("Testset Root Mean Squared Error(测试集均方根误差):",np.sqrt(np.mean(np.array(rmse_list)**2)) )
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=labels, mode='lines', name='TrueValue', line=dict(color='purple', width=5)))
    fig.add_trace(go.Scatter(y=results, mode='lines', name='Prediction', line=dict(color='gold', width=5)))

    fig.update_layout(title='Test State')

    fig.show()
    print("Testset R2(测试集拟合曲线决定系数):",r2_score(labels, results))
    return fig, r2_score(labels, results), np.mean(mae_list), np.sqrt(np.mean(np.array(rmse_list)**2))

# 检验模型拟合情况
def inspect_model_fit(model, train_loader, scaler):
    model = model
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, label in train_loader:
        pred = model(seq)
        pred = pred[:, 0, -1]
        label = label[:, 0, -1]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i])
            labels.append(label[i])

    r2 = r2_score(labels, results)
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=labels, mode='lines', name='History', line=dict(color='purple',width=5)))
    fig.add_trace(go.Scatter(y=results, mode='lines', name='Prediction', line=dict(color='gold', width=5)))

    fig.update_layout(title='Inspect model fit state')

    fig.show()
    print("Trainingset R2(训练集拟合曲线决定系数):", r2)
    return fig, r2

def calculate_mae(y_true, y_pred):
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_pred - y_true))
    return mae

def calculate_rmse(y_true, y_pred):
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse