from cleanfid import fid

# path to CelebA-HQ dataset
base_path = "celeba_hq_256"

test_paths = [
    "./samples"
]

fid_scores = []

for path in test_paths:
    score = fid.compute_fid(base_path, path)
    fid_scores.append((path, score))
    print(f"FID score for {path}: {score}")

output_file = "./fid_results.txt"
with open(output_file, 'w') as file:
    for path, score in fid_scores:
        file.write(f"{path}: {score}\n")

print(f"Results saved to {output_file}")