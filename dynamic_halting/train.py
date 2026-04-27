import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json

from dataset import DraftLogitsDataset
from model import DynamicHaltingMLP

def main():
    # Configuration
    data_path = Path(__file__).parent.parent / "data" / "data_for_MLP.csv"
    batch_size = 64
    epochs = 20
    learning_rate = 1e-3
    
    print(f"Loading dataset from {data_path}...")
    full_dataset = DraftLogitsDataset(data_path)
    
    # Train / Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    model = DynamicHaltingMLP(input_dim=2, hidden_dim=16)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                # Accuracy tracking
                predicted_labels = (predictions >= 0.5).float()
                correct += (predicted_labels == Y_batch).sum().item()
                total += Y_batch.size(0)
                
        val_loss /= len(val_dataset)
        val_accuracy = correct / total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}%")
            
    # Save the model
    output_dir = Path(__file__).parent.parent / "weights"
    model_path = output_dir / "mlp_weights.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete! Weights saved to {model_path}")
    
    # Save the normalisation params
    norm_params = full_dataset.get_norm_params()
    with open(output_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=4)
    print("Normalisation parameters saved to norm_params.json")

if __name__ == "__main__":
    main()
