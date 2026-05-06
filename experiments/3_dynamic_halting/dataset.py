import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DraftLogitsDataset(Dataset):
    def __init__(self, csv_file):
        """
        Loads the CSV file containing the draft features.
        Columns: entropy, max_prob, accepted
        """
        self.data = pd.read_csv(csv_file)
        
        # Features X: [entropy, max_prob]
        raw_X = self.data[['entropy', 'max_prob']].values.astype(np.float32)
        
        # Normalize the features for better MLP training
        # Saving mean and std to a dict so we can use it during inference
        self.mean = raw_X.mean(axis=0)
        self.std = raw_X.std(axis=0)
        
        # Avoid division by zero
        self.std[self.std == 0] = 1e-6
        
        self.X = (raw_X - self.mean) / self.std
        self.X = torch.tensor(self.X)
        
        # Labels Y: [accepted]
        self.Y = self.data['accepted'].values.astype(np.float32)
        self.Y = torch.tensor(self.Y).unsqueeze(1) # [N, 1] for BCELoss

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
        
    def get_norm_params(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
