import torch
from torch.utils.data import DataLoader
import argparse
import os

import config
from src.dataset import BreastCancerDataset, get_transforms
from src.models import UNetHybrid
from src.utils import check_accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained UNet hybrid model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument("--encoder", type=str, required=True, choices=["mobilenet", "densenet", "resnet"],
                        help="Encoder backbone used during training.")
    args = parser.parse_args()

    _, val_transform = get_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    model = UNetHybrid(encoder_name=args.encoder, in_channels=3, out_channels=1).to(config.DEVICE)
    
    # Load the trained model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    test_loader = DataLoader(
        BreastCancerDataset(
            image_dir=os.path.join(config.DATA_PATH, "test/images"),
            mask_dir=os.path.join(config.DATA_PATH, "test/masks"),
            transform=val_transform
        ),
        batch_size=1, # Evaluate one image at a time
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False
    )
    
    print(f"Evaluating model with {args.encoder} backbone...")
    check_accuracy(test_loader, model, device=config.DEVICE)

if __name__ == "__main__":
    main()