import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def calculate_clip_similarity(folder1, folder2):
    files1 = sorted(glob.glob(os.path.join(folder1, "*.png")) + 
                    glob.glob(os.path.join(folder1, "*.jpg")) + 
                    glob.glob(os.path.join(folder1, "*.jpeg")))
    files2 = sorted(glob.glob(os.path.join(folder2, "*.png")) + 
                    glob.glob(os.path.join(folder2, "*.jpg")) + 
                    glob.glob(os.path.join(folder2, "*.jpeg")))
    
    min_length = min(len(files1), len(files2))

    files1 = files1[:min_length]
    files2 = files2[:min_length]

    if len(files1) != len(files2):
        raise ValueError("The number of images in both folders should be the same.")

    cos = torch.nn.CosineSimilarity(dim=1)
    total_similarity = 0

    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Processing images"):
        try:
            # print(file1, "             ", file2)
            img1 = preprocess(Image.open(file1)).unsqueeze(0).to(device)
            img2 = preprocess(Image.open(file2)).unsqueeze(0).to(device)

            with torch.no_grad():
                features1 = model.encode_image(img1)
                features2 = model.encode_image(img2)

            similarity = cos(features1, features2).item()
            similarity = (similarity + 1) / 2

            print(similarity)
            total_similarity += similarity
        except Exception as e:
            print(f"Error processing images {file1} and {file2}: {e}")

    average_similarity = total_similarity / len(files1)
    return average_similarity

if __name__ == "__main__":
    folder1 = "./original_images"
    
    paths_list = [
        "./samples"
    ]
    
    for _path in paths_list:
        try:
            avg_similarity = calculate_clip_similarity(folder1, _path)
            print(_path)
            print(f"Average CLIP similarity score: {avg_similarity:.4f}")
        except Exception as e:
            print(f"An error occurred: {e}")