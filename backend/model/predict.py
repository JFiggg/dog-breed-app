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
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier = nn.Linear(self.model.classifier[1].in_features, len(self.class_map))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        self.model.load_state_dict(checkpoint)
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
        model_path='model/weights/model_epoch_30.pth',
        train_data_path='data/train'
    )
    
    # Test prediction
    breed = predictor.predict('data/train/n02096051-Airedale/n02096051_9359.jpg')
    print(f'The predicted breed is a {breed}')