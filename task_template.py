import os
import sys
import requests
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
import pandas as pd

# --------------------------------
# LOADING A MODEL (EXAMPLE: TARGET MODEL)
# --------------------------------

def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

checkpoint_path = "path/to/your/model_checkpoint.safetensors"  # Replace with your model checkpoint path 
state_dict = load_file(checkpoint_path, device="cpu")

model = make_model() 
model.load_state_dict(state_dict, strict=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

data_root = "path/to/cifar100"  # Replace with your CIFAR-100 dataset path, or where it should be downloaded
dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
x, y = dataset[0]  # Example: get the first image and label

with torch.no_grad():
    logits = model(x.unsqueeze(0))

print("True label:", y)
print("Logits shape:", logits.shape)  # Should be [1, 100] for CIFAR-100
print("Logits:", logits)

# # --------------------------------
# # SUBMISSION FORMAT
# # --------------------------------

"""
The submission must be a .csv file with the following format:

-"id": ID of the subset (from 0 to 359)
-"score": Stealing confidence score for each image (float)
"""

# Example Submission:

subset_ids = list(range(360))  
confidence_scores = torch.rand(len(subset_ids)).tolist()
submission_df = pd.DataFrame({
    "id": subset_ids,
    "score": confidence_scores
})
submission_df.to_csv("example_submission.csv", index=None)