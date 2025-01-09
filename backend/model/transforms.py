from torchvision import transforms

# Standard transformations for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip some images (data augmentation)
    transforms.RandomRotation(10),  # Slightly rotate some images
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# Simpler transformations for testing/validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])