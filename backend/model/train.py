import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from dataset import DogBreedDataset
from transforms import train_transform, val_transform
from tqdm import tqdm  # For progress bars

def train_model():
    print("Initializing training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = DogBreedDataset(
        root_dir='../data/train',
        transform=train_transform
    )
    val_dataset = DogBreedDataset(
        root_dir='../data/val',
        transform=val_transform
    )
    
    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load model
    print("Loading model...")
    model = models.resnet50(weights='IMAGENET1K_V1')  # Updated from pretrained=True
    num_classes = len(train_dataset.class_map)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training for {num_classes} classes...")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss/(batch_idx+1)})
            
        # Print epoch statistics
        avg_loss = running_loss/len(train_loader)
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}')
        
        # Save model
        print("Saving model checkpoint...")
        torch.save(model.state_dict(), f'weights/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")