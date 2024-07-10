import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import json
import psutil
import GPUtil
import subprocess
import time 
import os

#----------------- set seed ------------------
torch.manual_seed(42)
np.random.seed(42)

# Get the current script's file path
script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)
data_directory = parent_directory + '/data/'


def monitor_gpu():
    command = "nvidia-smi --query-gpu=power.draw,memory.total,memory.used --format=csv,noheader,nounits"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    gpu_data = output.strip().decode().split(', ')

    total_memory = float(gpu_data[1])
    used_memory = float(gpu_data[2])
    vram_usage = (used_memory / total_memory) 

    # Get GPU load using GPUtil
    gpu = GPUtil.getGPUs()[0]
    gpu_load = gpu.load 

    return gpu_load, vram_usage



def monitor_system():
    cpu_usage = (psutil.cpu_percent())/100

    ram_usage = (psutil.virtual_memory().percent)/100
    return cpu_usage, ram_usage



def hidden_blocks(input_size, output_size, activation_function):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activation_function,
    )


class MLP(nn.Module):
    def __init__(self, input_size=75, hidden_units=512, num_classes=10, activation_function=nn.LeakyReLU()):
        super(MLP, self).__init__()

        self.architecture = nn.Sequential(
            hidden_blocks(input_size, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        return self.architecture(x)


def main():
    # Initialize lists to store system metrics
    cpu_usage_list = []
    ram_usage_list = []
    gpu_load_list = []
    vram_usage_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Current device:", device)

    le = LabelEncoder()
    data = pd.read_csv(data_directory + 'dataset_v8.50.csv')

    X = data.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
    y = data['skill_id']

    # encode the labels
    y = le.fit_transform(y)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # splitting and grouping by video_name
    train_idx, test_idx = next(gss.split(X, y, groups=data['video_name']))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    input_size = len(data.columns) - 3  # exclude 'id_video', 'frame', 'skill_id'
    hidden_units = 2048
    num_classes = len(data['skill_id'].unique())
    lr = 0.0001
    n_epochs = 400
    batch_size = 512

    model = MLP(input_size, hidden_units, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=1e-6, weight_decay=1e-4)
    #optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=1e-4, momentum=0, centered=False)

    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for epoch in range(n_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % int(batch_size / 4) == 0:
                # Monitoring GPU load and VRAM
                gpu_load, vram_usage = monitor_gpu()
                gpu_load_list.append(gpu_load)
                vram_usage_list.append(vram_usage)

                # Monitoring system metrics
                cpu_usage, ram_usage = monitor_system()
                cpu_usage_list.append(cpu_usage) 
                ram_usage_list.append(ram_usage)

    #Performing evaluation
    end_time = time.time()
    training_time = end_time - start_time
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device), torch.LongTensor(y_test).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            
            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()

            gt_labels = labels.tolist()
            predicted_labels = predicted.tolist()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = loss.item()

    #`compute accuracy and f1 score
    accuracy = correct / total
    print('Accuracy: ', accuracy)

    # f1 score
    
    f1 = f1_score(gt_labels, predicted_labels, average='weighted')
    print('F1 score: ', f1)


    # Save system metrics to a JSON file
    metrics_data = {
        "cpu_usage": cpu_usage_list,
        "ram_usage": ram_usage_list,
        "gpu_load": gpu_load_list,
        "vram_usage": vram_usage_list, 
        "training_time": training_time,
        "accuracy": accuracy,
        "f1_score": f1
    }

    with open('enter/your/path', 'w') as f:
        json.dump(metrics_data, f, indent=4)

if __name__ == '__main__':
    main()
