import os
import json

def get_min_max_mean_from_file(file_path):
    min_max_mean_values = {
        "gpu_load": {"min": float('inf'), "max": float('-inf'), "sum": 0, "count": 0, "mean": 0},
        "vram_usage": {"min": float('inf'), "max": float('-inf'), "sum": 0, "count": 0, "mean": 0},
        "cpu_usage": {"min": float('inf'), "max": float('-inf'), "sum": 0, "count": 0, "mean": 0},
        "ram_usage": {"min": float('inf'), "max": float('-inf'), "sum": 0, "count": 0, "mean": 0}
    }

    with open(file_path, 'r') as f:
        data = json.load(f)
        update_min_max_mean(data, min_max_mean_values)

    for field in min_max_mean_values.keys():
        if min_max_mean_values[field]["count"] > 0:
            min_max_mean_values[field]["mean"] = min_max_mean_values[field]["sum"] / min_max_mean_values[field]["count"]

    return min_max_mean_values

def update_min_max_mean(data, min_max_mean_values):
    for field in min_max_mean_values.keys():
        if field in data:
            values = data[field]
            min_val = min(values)
            max_val = max(values)
            sum_val = sum(values)
            count_val = len(values)

            if min_val < min_max_mean_values[field]["min"]:
                min_max_mean_values[field]["min"] = min_val
            if max_val > min_max_mean_values[field]["max"]:
                min_max_mean_values[field]["max"] = max_val
            
            min_max_mean_values[field]["sum"] += sum_val
            min_max_mean_values[field]["count"] += count_val

#Usage

# if __name__ == "__main__":
#     folder = "optimizer"
#     folder_path = folder

#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.json'):
#             file_path = os.path.join(folder_path, file_name)
#             min_max_mean_values = get_min_max_mean_from_file(file_path)
            
#             result_file_name = f"{os.path.splitext(file_name)[0]}_minmaxmean.json"
#             result_path = os.path.join(folder_path, result_file_name)
            
#             with open(result_path, 'w') as f:
#                 json.dump(min_max_mean_values, f, indent=4)

