import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class StandardScaler():
    """Description: Normalize the data"""
    def __init__(self):
        self.mean = 0.
        self.std = 1.
 
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
 
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
 
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, feature):
    """
    Create training sequences and corresponding labels for time series data.

    Args:
    input_data (numpy.array or torch.Tensor): The input time series data.
    tw (int): The length of the training sequence.
    pre_len (int): The length of the prediction sequence.

    Returns:
    inout_seq (list): A list containing training sequences and corresponding labels. Each element is a tuple where the first element is the training sequence and the second element is the corresponding label.

    """
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def create_dataloader(data_path, pre_len, window_size, target, feature, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>Creating DataLoader<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(data_path)  
    pre_len = pre_len  
    train_window = window_size  

    # Move the feature column to the end
    target_data = df[[target]]
    df = df.drop(target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    true_data = df_data.values

    train_data = true_data[:int(0.7 * len(true_data))]
    valid_data = true_data[int(0.7 * len(true_data)):int(0.8 * len(true_data))]
    test_data = true_data[int(0.8 * len(true_data)):]

    print("Training set size:", len(train_data), "Test set size:", len(test_data), "Validation set size:", len(valid_data))
    
    # Define standardization optimizer
    scaler = StandardScaler()
    scaler.fit(train_data)

    # Perform standardization
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)

    # Convert to Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    # Define the input of the trainer
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, feature)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, feature)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, feature)

    # Create dataset
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, drop_last=True)

    print("Training set data:   ", len(train_inout_seq), "Converted to batch data:", len(train_loader))
    print("Test set data:       ", len(test_inout_seq), "Converted to batch data:", len(test_loader))
    print("Validation set data: ", len(valid_inout_seq), "Converted to batch data:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>DataLoader Created<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    output_string = (
        "Data processing is complete. "
        f"Training set data: {len(train_inout_seq)} Converted to batch data: {len(train_loader)};"
        f"Test set data: {len(test_inout_seq)} Converted to batch data: {len(test_loader)};"
        f"Validation set data: {len(valid_inout_seq)} Converted to batch data: {len(valid_loader)};"
    )
    return train_loader, test_loader, valid_loader, scaler, output_string
