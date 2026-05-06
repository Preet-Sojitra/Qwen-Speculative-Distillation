import torch.nn as nn

class DynamicHaltingMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        """
        A lightweight 2-layer MLP to predict whether the target model 
        will accept the draft token based on entropy and max_prob.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, 2]
        return self.net(x)
