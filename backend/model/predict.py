import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from transforms import val_transform
from dataset import DogBreedDataset

class DogBreedPredictor:
    def __init__(self, model_path, train_data_path):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class mapping from training data
        train_dataset = DogBreedDataset(train_data_path, transform=None)
        self.class_map = train_dataset.class_map
        
        # Load model
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_map))
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = val_transform(image)
        image = image.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            
            # Get predicted breed name
            for breed, idx in self.class_map.items():
                if idx == predicted.item():
                    return breed
        
        return None

if __name__ == '__main__':
    predictor = DogBreedPredictor(
        model_path='model/weights/model_epoch_10.pth',
        train_data_path='data/train'
    )
    
    # Test prediction
    breed = predictor.predict('path/to/test/image.jpg')
    print(f'Predicted breed: {breed}')