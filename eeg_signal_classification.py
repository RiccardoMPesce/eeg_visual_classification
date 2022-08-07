import torch
import torch.nn.functional as F

import platform
import importlib

from pathlib import Path

from torch.utils.data import DataLoader

print(f"Platform used: {platform.platform()}")

EEG_DATASET_PATH = "data/eeg_5_95_std.pth"

SPLITS_PATH = "data/block_splits_by_image_all.pth"

# Leave this always to zero
SPLIT_NUM = 0

# Subject selecting
# Choose a subject from 1 to 6, default is 0 (all subjects)
SUBJECT = 0

# Time options: select from 20 to 460 samples from EEG data
TIME_LOW = 20
TIME_HIGH = 460

# Model type/options
# Specify which generator should be used. Available: lstm | EEGChannelNet
# It is possible to test out multiple deep classifiers:
#   - lstm is the model described in the paper 
#     "Deep Learning Human Mind for Automated Visual Classification”, CVPR 2017
#   - model10 is the model described in the paper 
#     "Decoding brain representations by multimodal learning of neural activity and visual features", TPAMI 2020
MODEL_TYPE = "lstm"

MODEL_PARAMS = ""
PRETRAINED_NET = ""

# Training options
BATCH_SIZE = 16
OPTIMIZER = "Adam"
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_BY = 0.5
LEARNING_RATE_DECAY_EVERY = 10
DATA_WORKERS = 4
EPOCHS = 200

# Save every SAVE_CHECK epochs
SAVE_CHECK = 2

# Backend options
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("CUDA available")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("MPS (Metal) available")
else:
    DEVICE = torch.device("cpu")
    print("CPU available")

torch.utils.backcompat.broadcast_warning.enabled = True

# Force CPU
# print("Forcing CPU")
# DEVICE = torch.device("cpu")

# Debug Mode
DEBUG = True

class EEGDataset:
    def __init__(self, eeg_signals_path, subject=0, time_low=20, time_high=460, model_type="lstm"):
        self.subject = subject
        self.time_low = time_low
        self.time_high = time_high
        self.model_type = model_type

        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if subject != 0:
            self.data = [loaded["dataset"][i] for i in range(
                len(loaded["dataset"])) if loaded["dataset"][i]["subject"] == subject]
        else:
            self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)

    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high, :]

        if self.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, self.time_high - self.time_low)
        # Get label
        label = self.data[i]["label"]
        
        return eeg, label

class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [
            i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600
        ]
        # Compute size
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        
        return eeg, label


dataset = EEGDataset(EEG_DATASET_PATH)
loaders = {
    split: DataLoader(
        Splitter(
            dataset, 
            split_path=SPLITS_PATH, 
            split_num=SPLIT_NUM, 
            split_name=split),
            batch_size=BATCH_SIZE, 
            drop_last=True, 
            shuffle=True
    ) for split in ["train", "val", "test"]
}

# Load model
model_options = {
    key: int(value) if value.isdigit() else (
        float(value) if value[0].isdigit() else value
    ) for (key, value) in [x.split("=") for x in MODEL_PARAMS]
}

if DEBUG:
    print(f"Model options: {model_options}")

# Create discriminator model/optimizer
module = importlib.import_module("models." + MODEL_TYPE)
model = module.Model(**model_options)
# Moving model to the appropriate device
model.to(DEVICE)
print(f"Model has been moved to device '{DEVICE}'")
# Creating the optimizer
optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE)

if PRETRAINED_NET != "":
    print(f"Loading a pretrained model from '{PRETRAINED_NET}'")
    model = torch.load(PRETRAINED_NET)
    model.to(DEVICE)
    
if DEBUG:
    print(f"Model: {model}")

# Initialize training,validation, test losses and accuracy list
losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0

# Start training
predicted_labels = [] 
correct_labels = []

for epoch in range(1, EPOCHS + 1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    
    # Adjust learning rate for SGD
    if OPTIMIZER == "SGD":
        lr = LEARNING_RATE * (LEARNING_RATE_DECAY_BY ** (epoch // LEARNING_RATE_DECAY_EVERY))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]):
            # Move tensors to device
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            
            if DEBUG:
                print(f"Device is: {input.device}")
                print(f"Input shape is: {input.shape}")
                print(f"Target shape is: {target.shape}")

            # Forward
            output = model(input)

            if DEBUG:
                print(f"Output generated (shape {output.shape})")

            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()

            if DEBUG:
                print(f"Loss computed: {loss}")
                print(f"Loss device: {loss.device}")
            
            # Compute accuracy
            _, pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct / input.data.size(0)   
            accuracies[split] += accuracy
            counts[split] += 1

            if DEBUG:
                print(f"Pred array shape: {pred.shape}")
                print(f"Accuracy is: {accuracy}")
            
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                if DEBUG:
                    print("Zeroed grad")
                loss.backward()
                if DEBUG:
                    print("Calculated derivative")
                optimizer.step()
                if DEBUG:
                    print("Made optimizer step")

            if DEBUG:
                print(f"Backpropagation done: {loss}")
    
    # Print info at the end of the epoch
    if accuracies["val"] / counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"] / counts["val"]
        best_accuracy = accuracies["test"] / counts["test"]
        best_epoch = epoch

    train_loss = losses["train"] / counts["train"]
    train_accuracy = accuracies["train"] / counts["train"]
    validation_loss = losses["val"] / counts["val"]
    validation_accuracy = accuracies["val"] / counts["val"]
    test_loss = losses["test"] / counts["test"]
    test_accuracy = accuracies["test"] / counts["test"]

    print("\nINFO")
    print(f"- Model: {MODEL_TYPE}")
    print(f"- Subject: {SUBJECT}")
    print(f"- Time interval: [{TIME_LOW}-{TIME_HIGH}] [{TIME_LOW}-{TIME_HIGH} Hz]")
    print(f"- Epoch: {epoch}")
    print("\nSTATS")
    print(f"- Training: Loss {train_loss:.4f}, Accuracy {train_accuracy:.4f}")
    print(f"- Validation: Loss {validation_loss:.4f}, Accuracy {validation_accuracy:.4f}")
    print(f"- Test: Loss {test_loss:.4f}, Accuracy {test_accuracy:.4f}")
    print(f"Best Test Accuracy at maximum Validation Accuracy (validation_accuracy = {best_accuracy_val}) is {best_accuracy} at epoch {best_epoch}")

    losses_per_epoch["train"].append(train_loss)
    losses_per_epoch["val"].append(validation_loss)
    losses_per_epoch["test"].append(test_loss)
    accuracies_per_epoch["train"].append(train_accuracy)
    accuracies_per_epoch["val"].append(validation_accuracy)
    accuracies_per_epoch["test"].append(test_accuracy)

    if epoch % SAVE_CHECK == 0:
        torch.save(model, f"{MODEL_TYPE}__subject{SUBJECT}_epoch_{epoch}.pth")
