import torch
import coremltools as ct
from torchvision import models
import torch.nn as nn

# First install coremltools
# pip install coremltools

def convert_model():
    # Load your trained model
    model = models.resnet50()
    num_classes = 120  # Number of dog breeds
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load your trained weights
    model.load_state_dict(torch.load('weights/best_model.pth'))
    model.eval()
    
    # Trace the model with example input
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input_1", shape=example_input.shape)]
    )
    
    # Save the model
    mlmodel.save("DogBreedClassifier.mlmodel")
    print("Model converted and saved as DogBreedClassifier.mlmodel")

if __name__ == "__main__":
    convert_model()