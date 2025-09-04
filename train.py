import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

import config
from src.dataset import BreastCancerDataset, get_transforms
from src.models import UNetHybrid
from src.utils import save_checkpoint, load_checkpoint, check_accuracy

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    parser = argparse.ArgumentParser(description="Train a UNet hybrid model.")
    parser.add_argument("--encoder", type=str, required=True, choices=["mobilenet", "resnet"],
                        help="Encoder backbone to use.")
    args = parser.parse_args()

    train_transform, val_transform = get_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    model = UNetHybrid(encoder_name=args.encoder, in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader = DataLoader(
        BreastCancerDataset(
            image_dir=os.path.join(config.DATA_PATH, "train/images"),
            mask_dir=os.path.join(config.DATA_PATH, "train/masks"),
            transform=train_transform
        ),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(os.path.join(config.OUTPUT_PATH, f"unet_{args.encoder}.pth")), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        output_filename = os.path.join(config.OUTPUT_PATH, f"unet_{args.encoder}.pth")
        os.makedirs(config.OUTPUT_PATH, exist_ok=True)
        save_checkpoint(checkpoint, filename=output_filename)
        
        # Check accuracy on a portion of training data as an example
        check_accuracy(train_loader, model, device=config.DEVICE)


if __name__ == "__main__":
    main()
