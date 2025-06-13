import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings("ignore")

def gender_classifier(image_paths):
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

    all_gender_outputs = []
    gender_outputs_all = 0.0
    
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0).to("cuda:0")  # Add batch dimension and move to GPU

            # Get model outputs
            outputs = fairface(image)
            gender_outputs = outputs[:, 7:9]
            gender_outputs = torch.nn.functional.softmax(gender_outputs, dim=1)

            all_gender_outputs.append(gender_outputs.cpu().numpy())

            gender_outputs_all += gender_outputs.sum(dim=0)

    # Stack all outputs into a single numpy array
    all_gender_outputs = np.vstack(all_gender_outputs)
    
    # Print statistics
    male_pro = np.sum(all_gender_outputs[:, 0] > 0.5) / all_gender_outputs.shape[0]
    female_pro = np.sum(all_gender_outputs[:, 1] > 0.5) / all_gender_outputs.shape[0]
    print("male: ", np.sum(all_gender_outputs[:, 0] > 0.5) / all_gender_outputs.shape[0],
          "female: ", np.sum(all_gender_outputs[:, 1] > 0.5) / all_gender_outputs.shape[0],
          "no face: ", (all_gender_outputs.shape[0] - np.sum(all_gender_outputs[:, 0] > 0.75) - np.sum(all_gender_outputs[:, 1] > 0.75)) / all_gender_outputs.shape[0])
    
    print("FD (logits mean method): ", np.sqrt((gender_outputs_all[0].cpu().numpy()/len(image_paths) - 0.5)**2 + (gender_outputs_all[1].cpu().numpy()/len(image_paths) - 0.5)**2))
    return all_gender_outputs


if __name__ == "__main__":
    file_lists = ["./samples"]

    output_file = "./gender_results.txt"

    for _file in file_lists:
        print("File Path: ", _file)
        image_paths = glob.glob(os.path.join(_file, "*.png"))
        image_paths = sorted(image_paths)

        results = gender_classifier(image_paths)

        with open(output_file, 'a') as file:
            for idx, (path, result) in enumerate(zip(image_paths, results)):
                file.write(f"Image {os.path.split(path)[-1]}: male: {result[0]}, female: {result[1]}\n")

        print(f"Results saved to {output_file}")
