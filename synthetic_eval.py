import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import Mamba
from tqdm import tqdm
from synthetic_config import MambaConfig, training_config, copy_dataset_config
from copy_generator import generate_copying_dataset as generate_dataset # TODO make more general for induction task

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Define model
mambaconfig = MambaConfig()
model = Mamba(mambaconfig).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

# Training function
def train(dataset_config):
    """
    Train the model.
    """
    model.train()
    start_time = time.time()

    for step in tqdm(range(training_config["num_steps"]), desc = "Training synthetic task"):
        step_loss = 0
        correct = 0
        total = 0
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(1) == targets).sum().item()
        accuracy = 100 * correct / total
        print(f'Step [{step+1}/{training_config["num_steps"]}], Loss: {step_loss/training_config["batch_size"]:.4f}, Accuracy: {accuracy:.2f}%')

    end_time = time.time()
    print(f'Training completed in: {(end_time - start_time)/60:.2f} minutes')

# Validation function
def validate(dataset_config):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        print(f"{targets.shape = }", f"{outputs.shape = }")
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(1) == targets).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy


'''
Script to run:
train(copy_dataset_config)
validate(copy_dataset_config)
'''
