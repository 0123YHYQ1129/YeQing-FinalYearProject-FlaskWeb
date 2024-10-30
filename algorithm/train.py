import time
import torch
from tqdm import tqdm
import torch.nn as nn
import plotly.graph_objects as go
import numpy as np

def train(model, feature, train_loader, scaler):
    model = model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = 20
    model.train()  # train mode
    results_loss = []
    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad() 
            y_pred = model(seq)
            if feature == 'MS':
                y_pred = y_pred[:, :, -1].unsqueeze(2)
            single_loss = loss_function(y_pred, labels)
 
            single_loss.backward()
 
            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
#        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss))
 
 
        torch.save(model.state_dict(), "model.pth")
        time.sleep(0.1)
    return plot_loss_data(results_loss)

def plot_loss_data(data):
    # Draw the curve by plotly
    fig = go.Figure()

    # Change the color of the line to gold
    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Loss', line=dict(color='gold',width = 5)))

    # Find the minimum point
    min_val = np.min(data)
    min_idx = np.argmin(data)

    # Add a marker for the minimum point
    fig.add_trace(go.Scatter(x=[min_idx], y=[min_val], mode='markers', 
                             marker=dict(color='purple', size=10), 
                             text=f"Min Loss: {min_val} at {min_idx}", 
                             name='Min Loss'))

    # Add title
    fig.update_layout(title='Loss Results Plot',
                      xaxis=dict(showgrid=True, gridwidth=1, gridcolor='Purple'),
                      yaxis=dict(showgrid=True, gridwidth=1, gridcolor='Purple'))

    fig.show()

    return fig