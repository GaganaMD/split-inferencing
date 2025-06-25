# import torch

# # Load the YOLOv5s model from the Ultralytics repo
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Print the model architecture (all layers)
# print(model)

import torch
import os
from collections import Counter

# Handle __file__ fallback for Jupyter or interactive environments
try:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    script_name = "interactive_run"

# Create output directory
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
file_path = os.path.join(result_dir, f"{script_name}.txt")

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Utility: count layer types
def count_layer_types(module):
    return Counter(type(layer).__name__ for layer in module.modules())

# Collect frequencies
freq_model = count_layer_types(model)
freq_model_model = count_layer_types(model.model)
try:
    freq_model_model_model = count_layer_types(model.model.model)
except AttributeError:
    freq_model_model_model = "model.model.model does not exist"

# Write to file
with open(file_path, "w") as f:
    f.write("Layer type frequency in model:\n")
    for layer_type, count in freq_model.items():
        f.write(f"  {layer_type}: {count}\n")

    f.write("\nLayer type frequency in model.model:\n")
    for layer_type, count in freq_model_model.items():
        f.write(f"  {layer_type}: {count}\n")

    f.write("\nLayer type frequency in model.model.model:\n")
    if isinstance(freq_model_model_model, str):
        f.write(freq_model_model_model + "\n")
    else:
        for layer_type, count in freq_model_model_model.items():
            f.write(f"  {layer_type}: {count}\n")

print(f"Layer frequencies saved to: {file_path}")
