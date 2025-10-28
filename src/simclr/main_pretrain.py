import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# --- Import our custom and project-level components ---
# Data-related imports
import src.data.flat_data as flat_data
from simclr.data import SimCLRTransform, simclr_collate # Our new data components

# Model-related imports
from flat_mae.models_mae import MaskedViT # Reusing the backbone
from simclr.models import ContrastiveModel # Our new main model

def main():
    """
    Main function to run a simplified contrastive pre-training loop.
    Supports both 'simclr' and 'simsiam' modes.
    """
    print("--- Starting Contrastive Pre-training Script ---")

    # --- 1. Configuration ---
    # File Paths
    data_folder_path = Path("nsd-train-task-clips-16t") # Assumes this is in the project root

    # Model Hyperparameters
    CONTRASTIVE_MODE = "simclr"  # <-- CHANGE THIS TO "simsiam" TO TEST THE OTHER MODE
    BACKBONE_EMBED_DIM = 384     # For ViT-Small

    # Training Hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5 # Run for a few epochs for verification
    MASK_RATIO = 0.75 # Ratio of patches to mask in the encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running in mode: '{CONTRASTIVE_MODE}'")

    # --- 2. Data Pipeline Setup ---
    print("\n--- Setting up Data Pipeline ---")
    # Base transform with stochastic augmentations (e.g., random crop)
    base_transform = flat_data.make_flat_transform(
        img_size=(224, 560),
        normalize='global',
        random_crop=True,
        crop_kwargs={'scale': (0.8, 1.0), 'ratio': (2.4, 2.6)}
    )
    # Wrap it to produce two views
    simclr_transform = SimCLRTransform(base_transform)
    
    dataset = flat_data.FlatClipsDataset(root=data_folder_path, transform=simclr_transform)
    
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=simclr_collate, # Use our custom collate function
        shuffle=True,
        num_workers=0 
    )
    print("Data pipeline setup complete.")

    # --- 3. Model and Optimizer Setup ---
    print("\n--- Setting up Model and Optimizer ---")
    # Initialize the backbone encoder
    backbone = MaskedViT(
        img_size=(224, 560),
        in_chans=1,
        embed_dim=BACKBONE_EMBED_DIM,
        depth=12,
        num_heads=6
    )
    
    # Initialize our main ContrastiveModel
    model = ContrastiveModel(
        backbone=backbone,
        mode=CONTRASTIVE_MODE,
        embed_dim=BACKBONE_EMBED_DIM
    ).to(device)
    
    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Model and optimizer setup complete.")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # --- 4. The Training Loop ---
    print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")
    
    model.train() # Set the model to training mode
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=False)
        total_loss = 0.0

        for batch_view_1, batch_view_2 in loop:
            # Move data to the configured device
            view_1_images = batch_view_1['image'].to(device)
            view_2_images = batch_view_2['image'].to(device)

            # --- Core Training Steps ---
            optimizer.zero_grad()
            loss = model(view_1_images, view_2_images, mask_ratio=MASK_RATIO)
            loss.backward()
            optimizer.step()
            # -------------------------

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Loss: {avg_loss:.4f}")

    print("\n--- Training Verification Complete ---")
    print("The script successfully ran a few training epochs without crashing.")


if __name__ == "__main__":
    main()