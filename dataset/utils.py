import torch

class EEGDataset:
    # Constructor
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

    # Get size
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
        # Return
        return eeg, label


# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <=
                          self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label
