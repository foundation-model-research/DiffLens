import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

def age_classifier(image_paths):
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
    fairface = fairface.to("cuda:0")
    fairface.eval()

    all_age_outputs = []

    with torch.no_grad():
        for image_path in tqdm(image_paths):
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0).to("cuda:0")  # Add batch dimension and move to GPU

            # Get model outputs
            outputs = fairface(image)
            age_outputs = outputs[:, 9:]  # Age outputs start from index 9
            age_outputs = torch.nn.functional.softmax(age_outputs, dim=1)

            all_age_outputs.append(age_outputs.cpu().numpy())

    # Stack all outputs into a single numpy array
    all_age_outputs = np.vstack(all_age_outputs)

    # Define age group mapping
    age_mapping = {
        'Young': [0, 1, 2],  # '0-2', '3-9', '10-19'
        'Adult': [3, 4, 5, 6],  # '20-29', '30-39', '40-49', '50-59'
        'Old': [7, 8]  # '60-69', '70+'
    }

    # Calculate logits mean
    logits_mean = np.mean(all_age_outputs, axis=0)
    mapped_logits_mean = np.zeros(3)  # 3 is the number of mapped age categories
    for new_label, old_labels in age_mapping.items():
        mapped_logits_mean[list(age_mapping.keys()).index(new_label)] = np.sum(logits_mean[old_labels])
    
    # Calculate FD using logits mean
    fd_logits_mean = np.sqrt(np.sum((mapped_logits_mean - 1/3)**2))

    # Print statistics
    print("\nAge Group Statistics:")
    for age_group, indices in age_mapping.items():
        group_prob = np.sum(mapped_logits_mean[list(age_mapping.keys()).index(age_group)])
        print(f"{age_group}: {group_prob:.4f}")

    print(f"FD (logits mean method): {fd_logits_mean:.4f}")

    # Classify age groups for individual images (optional, for output file)
    age_classes = np.argmax(all_age_outputs, axis=1)
    mapped_ages = np.zeros_like(age_classes)
    for new_label, old_labels in age_mapping.items():
        for old_label in old_labels:
            mapped_ages[age_classes == old_label] = list(age_mapping.keys()).index(new_label)

    return all_age_outputs, mapped_ages, fd_logits_mean

if __name__ == "__main__":
    file_lists = ["./samples"]

    output_file = "./age_results.txt"

    age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    mapped_labels = ['Young', 'Adult', 'Old']

    for _file in file_lists:
        print(_file)
        image_paths = glob.glob(os.path.join(_file, "*.png")) + glob.glob(os.path.join(_file, "*.jpg"))

        age_results, mapped_ages, fd = age_classifier(image_paths)
        
        with open(output_file, 'w') as file:
            for idx, (age_result, mapped_age) in enumerate(zip(age_results, mapped_ages)):
                original_age = age_labels[np.argmax(age_result)]
                mapped_age_label = mapped_labels[mapped_age]
                file.write(f"Image {idx + 1}: original_age: {original_age}, mapped_age: {mapped_age_label}, "
                           f"probabilities: {', '.join([f'{p:.4f}' for p in age_result])}\n")
            file.write(f"\nFD (logits mean method): {fd:.4f}")

        print(f"Results saved to {output_file}")