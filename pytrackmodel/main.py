from model_manager import ModelManager
import torch

# remove the model directory if it exists
import shutil
shutil.rmtree("models", ignore_errors=True)

# Create a PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
    torch.nn.Sigmoid()
)

# Create a model manager object
model_manager = ModelManager(model, "my_model")

model_manager.save()