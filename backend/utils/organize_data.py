import os
import shutil
from sklearn.model_selection import train_test_split
import random

def organize_dataset(raw_dir, train_dir, val_dir, test_dir):
    # Get all breed directories
    breed_dirs = os.listdir(os.path.join(raw_dir, 'Images'))
    
    for breed in breed_dirs:
        # Create breed directories in train/val/test
        os.makedirs(os.path.join(train_dir, breed), exist_ok=True)
        os.makedirs(os.path.join(val_dir, breed), exist_ok=True)
        os.makedirs(os.path.join(test_dir, breed), exist_ok=True)
        
        # Get all images for this breed
        images = os.listdir(os.path.join(raw_dir, 'Images', breed))
        
        # Split into train (70%), validation (15%), and test (15%)
        train_imgs, test_val_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(test_val_imgs, test_size=0.5, random_state=42)
        
        # Move files to respective directories
        for img in train_imgs:
            shutil.copy(
                os.path.join(raw_dir, 'Images', breed, img),
                os.path.join(train_dir, breed, img)
            )
        
        for img in val_imgs:
            shutil.copy(
                os.path.join(raw_dir, 'Images', breed, img),
                os.path.join(val_dir, breed, img)
            )
            
        for img in test_imgs:
            shutil.copy(
                os.path.join(raw_dir, 'Images', breed, img),
                os.path.join(test_dir, breed, img)
            )
        
        print(f'Processed {breed}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test')

if __name__ == '__main__':
    organize_dataset(
        raw_dir='raw_dataset',
        train_dir='train',
        val_dir='val',          
        test_dir='test'         
    )