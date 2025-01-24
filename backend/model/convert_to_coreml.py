import os
import torch
import coremltools as ct
from torchvision import models
import torch.nn as nn

def convert_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'weights', 'model_epoch_30.pth')
    
    print(f"Looking for model at: {weights_path}")
    
    # Load your trained model
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_classes = 120
    model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)

    # Load state
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
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
    output_path = os.path.join(current_dir, 'DogBreedClassifier.mlmodel')
    mlmodel.save(output_path)
    print("Model converted and saved as DogBreedClassifier.mlmodel")

if __name__ == "__main__":
    convert_model()