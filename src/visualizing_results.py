import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder_path = 'width'


data = {
    'cpu_usage': {},
    'ram_usage': {},
    'gpu_load': {},
    'vram_usage': {}
}

output_path = os.path.join(folder_path, "output_plot")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Read all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        activation_function = filename.split('.')[0]
        with open(os.path.join(folder_path, filename), 'r') as f:
            json_data = json.load(f)
            for key in data.keys():
                if key not in json_data:
                    continue
                data[key][activation_function] = json_data[key]


def smooth(values, window_size):
    if len(values) < window_size:
        raise ValueError("The window size must be less than or equal to the length of the values array.")
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

def plot_data(data, metric, window_size=100): # Default window size is 100
    plt.figure(figsize=(12, 6))
    for activation_function, values in data[metric].items():
        if values:  
            try:
                smoothed_values = smooth(values, window_size)
                plt.plot(smoothed_values, label=activation_function)
            except ValueError as e:
                print(f"Skipping {activation_function} for {metric} due to error: {e}")
        else:
            print(f"Skipping {activation_function} for {metric} due to empty values")
    plt.title(f'Trend of {metric} for different widths')
    plt.xlabel('Iterations/Index')
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'{metric}.png'))

for metric in data.keys():
    plot_data(data, metric)
