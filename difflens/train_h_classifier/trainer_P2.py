from torch import nn
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import os
import tqdm

from .data_P2 import AttributeLatentDataset, fast_train_test_split

from simple_parsing import Serializable

import yaml

H_SPACE_EXAMPLE = torch.zeros((1, 512, 8, 8))

class ClassifierTrainConfig(Serializable):
    latents_dataset_path: str = None
    """Your latents saved path"""

    train_batch_size: int = 32
    """train batch size"""

    target_attr: str = "gender"
    """Choose from ['gender', 'age', 'race']"""

    data_seed: int = 0
    """Load data seed."""

    checkpoint_dir: str = None
    """Trained checkpoint."""

    checkpoint_dir_to_save: str = None
    """Checkpoint save path."""

    epochs: int = 3
    """Training epochs."""

    learning_rate: float = 0.01
    """Classifier learning rate."""

    def update(self, other_dict):
        if other_dict is None:
            return
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)


class HiddenLinear(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(49)])
    
    def forward(self,x, t):
        x = x.reshape(x.shape[0],-1)
        x = self.linears[t](x)
        return x
    
def make_model(path):
    model = HiddenLinear().cuda()
    model.load_state_dict(torch.load(path, map_location='cuda:0'))
    model.eval()

    return model

def train(model, train_loader, val_loader, epochs=10, lr=0.001, model_save_path='model_checkpoint.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_accuracy = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (latents, labels) in enumerate(tqdm.tqdm(train_loader)):
            latents, labels = latents.cuda().to(torch.float32), labels.cuda()

            for t in range(49):
                optimizer.zero_grad()
                outputs = model(latents[:, t], t)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for latents, labels in val_loader:
                latents, labels = latents.cuda().to(torch.float32), labels.cuda()

                for t in range(49):
                    outputs = model(latents[:, t], t)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Save the model checkpoint if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation accuracy: {val_accuracy:.2f}%')

def test(model, test_loader):
    model.eval()
    all_preds = [[] for _ in range(49)]
    all_labels = []

    with torch.no_grad():
        for latents, labels in tqdm.tqdm(test_loader):
            latents, labels = latents.cuda().to(torch.float32), labels.cuda()

            for t in range(49):
                outputs = model(latents[:, t], t)
                _, predicted = outputs.max(1)
                all_preds[t].extend(predicted.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    accuracies = []

    return accuracies, all_preds, all_labels

def train_main_P2(main_args):
    args = ClassifierTrainConfig()

    with open(main_args.train_h_classifier.h_classifier_config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    args.update(cfg_dict)

    data_path = args.latents_dataset_path

    dataset = AttributeLatentDataset(data_path, args)
    print("split train and test")
    train_data, val_data = fast_train_test_split(dataset, test_size=0.1, random_state=args.data_seed)

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.train_batch_size, shuffle=False)
    
    # we only train on h_space
    _, c, h, w = H_SPACE_EXAMPLE.shape
    
    output_dim = len(dataset.categories)

    model = HiddenLinear(input_dim=int(c*h*w), output_dim=output_dim).cuda()

    # Load Checkpoint
    if hasattr(args, "checkpoint_dir") and args.checkpoint_dir is not None:
        model.load_state_dict(torch.load(args.checkpoint_dir, map_location='cuda:0'))

    checkpoint_dir_to_save = args.checkpoint_dir_to_save

    checkpoint_path = os.path.join(checkpoint_dir_to_save, f'data_seed_{args.data_seed}')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # best save path
    best_model_save_path = os.path.join(checkpoint_path, f'best_{args.target_attr}.pth')

    train(model, train_loader, val_loader, 
        epochs=args.epochs, lr = args.learning_rate,
        model_save_path=best_model_save_path,)

    test_accuracies, all_preds, all_labels = test(model, val_loader)

    return 