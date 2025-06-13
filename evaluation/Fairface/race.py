import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

def race_classifier(image_paths):
    # FairFace Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # FairFace classifier
    fairface_model_name_or_path = "./res34_fair_align_multi_7_20190809.pt"
    fairface = models.resnet34(pretrained=False)
    fairface.fc = nn.Linear(fairface.fc.in_features, 18)
    fairface.load_state_dict(torch.load(fairface_model_name_or_path))
    fairface = fairface.to("cuda:0").to(torch.float32)
    fairface.eval()

    all_race_outputs = []
    gender_outputs_all = 0.0
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0).to("cuda:0").to(torch.float32)  # Add batch dimension and move to GPU

            # Get model outputs
            outputs = fairface(image)
            race_outputs = outputs[:, :7]  # First 7 outputs correspond to race
            race_outputs = torch.nn.functional.softmax(race_outputs, dim=1)

            all_race_outputs.append(race_outputs.cpu().numpy())
            gender_outputs_all += race_outputs.sum(dim=0)

    # Stack all outputs into a single numpy array
    all_race_outputs = np.vstack(all_race_outputs)

    # Define race group mapping
    race_mapping = {
        'White': [0],
        'Black': [1],
        'Asian': [3, 4],  # East Asian and Southeast Asian
        'Indian': [5]
    }

    # Classify race groups
    race_classes = np.argmax(all_race_outputs, axis=1)
    mapped_races = np.zeros_like(race_classes)
    for new_label, old_labels in race_mapping.items():
        for old_label in old_labels:
            mapped_races[race_classes == old_label] = list(race_mapping.keys()).index(new_label)

    # Count occurrences
    race_counts = {race: np.sum(mapped_races == i) for i, race in enumerate(race_mapping.keys())}
    total_count = len(mapped_races)

    # Print statistics
    print("\nRace Group Statistics:")
    for race, count in race_counts.items():
        print(f"{race}: {count / total_count:.4f}")
    
    print("FD (logits mean method): ", np.sqrt((gender_outputs_all[0].cpu().numpy()/len(image_paths) - 0.25)**2 + (gender_outputs_all[1].cpu().numpy()/len(image_paths) - 0.25)**2 + (gender_outputs_all[2].cpu().numpy()/len(image_paths) - 0.25)**2 + (gender_outputs_all[3].cpu().numpy()/len(image_paths) - 0.25)**2))
    return all_race_outputs, mapped_races

if __name__ == "__main__":
    file_lists = ["./samples"]

    output_file = "./race_results.txt"

    original_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
    mapped_labels = ['White', 'Black', 'Asian', 'Indian']

    for _file in file_lists:
        print(_file)
        image_paths = glob.glob(os.path.join(_file, "*.png")) + glob.glob(os.path.join(_file, "*.jpg"))
        image_paths = sorted(image_paths)
        race_results, mapped_races = race_classifier(image_paths)
        
        with open(output_file, 'w') as file:
            for idx, (race_result, mapped_race) in enumerate(zip(race_results, mapped_races)):
                original_race = original_labels[np.argmax(race_result)]
                mapped_race_label = mapped_labels[mapped_race]
                file.write(f"Image {os.path.split(image_paths[idx])[-1]}: original_race: {original_race}, mapped_race: {mapped_race_label}, "
                           f"probabilities: {', '.join([f'{p:.4f}' for p in race_result])}\n")

        print(f"Results saved to {output_file}")