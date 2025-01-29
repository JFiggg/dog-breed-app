from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from transforms import train_transform, val_transform

class DogBreedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.file_list = []
        self.class_map = {}
        
        # Create class mapping and file list
        for class_idx, breed in enumerate(self.classes):
            self.class_map[breed] = class_idx
            breed_path = os.path.join(root_dir, breed)
            for img_name in os.listdir(breed_path):
                self.file_list.append((os.path.join(breed_path, img_name), class_idx))
    
    def __len__(self):
        # Returns total number of images
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load image and class
        img_path, class_idx = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx

    def get_class_name(self, class_idx):
        # Helper to get breed name from class index
        for breed, idx in self.class_map.items():
            if idx == class_idx:
                return breed
        return None

# Example usage:
if __name__ == "__main__":

    dataset = DogBreedDataset(
        root_dir="data/train",
        transform=train_transform
    )
    
    # Print some info about the dataset
    print(f"Total images: {len(dataset)}")
    print("\nClass mapping:")
    for breed, idx in dataset.class_map.items():
        print(f"{breed}: {idx}")