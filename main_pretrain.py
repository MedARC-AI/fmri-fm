# src/simclr/main_pretrain.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# --- Import components using the new modular structure ---
import src.data.flat_data as flat_data
import src.simclr.data as simclr_data
import src.flat_mae.models_mae as models_mae
import src.simclr.models as simclr_models
import src.simclr.loss as simclr_loss

def main():
    """
    Main function to run a simplified SimCLR pre-training loop.
    """
    print("--- Starting SimCLR Pre-training Verification Script ---")

    # --- 1. Configuration ---
    # File Paths
    # IMPORTANT: Paths are now relative to the project root, not the script location.
    data_folder_path = Path("nsd-train-task-clips-16t") # Assuming this folder is in the project root
    
    # Model Hyperparameters
    backbone_embed_dim = 384  # For ViT-Small
    projection_hidden_dim = 512
    projection_output_dim = 128 # As used in the SimCLR paper
    
    # Training Hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    temperature = 0.5
    num_epochs = 3 # Run for a few epochs to see it work

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Pipeline Setup ---
    print("\n--- Setting up Data Pipeline ---")
    base_transform = flat_data.make_flat_transform(
        img_size=(224, 560),
        normalize='global',
        random_crop=True,
        crop_kwargs={'scale': (0.8, 1.0), 'ratio': (2.4, 2.6)}
    )
    transform = simclr_data.SimCLRTransform(base_transform)
    dataset = flat_data.FlatClipsDataset(root=data_folder_path, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=simclr_data.simclr_collate,
        shuffle=True,
        num_workers=0
    )
    print("Data pipeline setup complete.")

    # --- 3. Model and Optimizer Setup ---
    print("\n--- Setting up Model and Optimizer ---")
    # Base Encoder (Backbone)
    backbone = models_mae.mae_vit_small(img_size=(224, 560), in_chans=1)
    
    # Projection Head
    projection_head = simclr_models.ProjectionHead(
        input_dim=backbone_embed_dim,
        hidden_dim=projection_hidden_dim,
        output_dim=projection_output_dim
    )
    
    # Full SimCLR Model
    model = simclr_models.SimCLRModel(backbone, projection_head).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss Function
    criterion = simclr_loss.NTXentLoss(temperature=temperature).to(device)
    
    print("Model, optimizer, and loss function setup complete.")

    # --- 4. The Training Loop ---
    print(f"\n--- Starting Training for {num_epochs} Epochs ---")
    
    for epoch in range(num_epochs):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        total_loss = 0.0

        for batch_view_1, batch_view_2 in loop:
            view_1_images = batch_view_1['image'].to(device)
            view_2_images = batch_view_2['image'].to(device)

            optimizer.zero_grad()
            z1, z2 = model(view_1_images, view_2_images)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")

    print("\n--- Training Verification Complete ---")
    print("Refactoring successful. The script ran correctly from its new location.")

if __name__ == "__main__":
    main()