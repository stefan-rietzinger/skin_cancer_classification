# thanks to https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

import sys,os
import optuna 
import optuna.visualization as vis
from optuna.trial import TrialState 
import torch 
import kagglehub
import lib.lib as lib
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor,Lambda
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.utils import resample
from tqdm import tqdm
import time
from torchsummary import summary

# INITIALIZE VARIABLES
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mos" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")
BATCHSIZE = 128
CLASSES = 2
DIR = os.path.join(os.getcwd(),"search")
EPOCHS = 20
N_TRAIN_EXAMPLES = BATCHSIZE * 100   # Limit training samples 
N_VALID_EXAMPLES = BATCHSIZE * 100   # Limit validation samples 

# class to import custom dataset
class MNIST_skin_cancer(Dataset):
    def __init__(self, data_path:str, dataset_type:str, test_size:float=0.2, valid_size:float=0.25, random_state:int=1, transform=None, target_transform=None):
        self.data_path = data_path
        self.img_dirs = [os.path.join(data_path,image_dirs) for image_dirs in ["ham10000_images_part_1","ham10000_images_part_2"]]
        self.transform = transform 
        self.target_transform = target_transform
        
        # split data in test,validation and train data:
        metadata = pd.read_csv(os.path.join(data_path,"HAM10000_metadata.csv"))
        train_data, test_data = train_test_split(metadata, test_size=test_size, random_state=random_state, stratify=metadata["dx"])
        train_data, valid_data = train_test_split(train_data, test_size=valid_size, random_state=random_state, stratify=train_data["dx"])

        # define binary mapping
        self.class_to_idx = {
            'akiec': 1,  # Cancerous
            'bcc': 1,    # Cancerous
            'mel': 1,    # Cancerous
            'df': 0,     # Non-cancerous
            'bkl': 0,    # Non-cancerous
            'vasc': 0,   # Non-cancerous
            'nv': 0      # Non-cancerous
        }

        # assign train, validation or test set to metadata!
        if dataset_type == "train":
            metadata = train_data
            self.metadata = self._undersample(metadata)
        elif dataset_type == "val":
            metadata = valid_data 
            self.metadata = self._undersample(metadata)
        elif dataset_type == "test":
            metadata = test_data
            self.metadata = self._undersample(metadata)
        else:
            raise ValueError("Invalid type for 'dataset_type'. Please choose from 'train','val' or 'test'!")
        
        # create dictionary to encode targets into integer values!
        # self.class_to_idx = {label: idx for idx, label in enumerate(self.metadata['dx'].unique())}

    def _undersample(self, metadata, random_state=42):
        """
        return dataset for which undersampling was performed!
        """
        # Map labels to binary (cancerous/non-cancerous)
        metadata['binary_label'] = metadata['dx'].map(self.class_to_idx)
        
        # Split into cancerous and non-cancerous groups
        cancerous_metadata = metadata[metadata['binary_label'] == 1]
        non_cancerous_metadata = metadata[metadata['binary_label'] == 0]

        # Determine the smaller group size
        min_count = min(len(cancerous_metadata), len(non_cancerous_metadata))

        # Resample both groups to have the same size
        balanced_cancerous = resample(cancerous_metadata, replace=False, n_samples=min_count, random_state=random_state)
        balanced_non_cancerous = resample(non_cancerous_metadata, replace=False, n_samples=min_count, random_state=random_state)

        # Combine both groups into a single balanced dataset
        balanced_metadata = pd.concat([balanced_cancerous, balanced_non_cancerous])

        return balanced_metadata

    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self,idx:int) -> tuple:
        # read out label and image name from metadata!
        label = self.metadata.iloc[idx].dx
        image_name = f"{self.metadata.iloc[idx].image_id}.jpg"

        # find image file in img_dirs and read into image!
        for directory in self.img_dirs:
            img_path = os.path.join(self.data_path, directory, image_name)
            if os.path.exists(img_path):
                image = read_image(img_path).float()
                break
        else:
            raise Exception("FILE DOES NOT EXIST!")
        
        # transform label into numerical value
        label = self.class_to_idx[label]
        
        # apply transform or target transform if wanted
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label    

def define_model(trial):
    # Parameters for convolutional layers
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 7)  # Number of Conv layers
    conv_filter_sizes = [trial.suggest_int(f"conv_filters_{i+1}", 16, 64, step=16) for i in range(num_conv_layers)]
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])  # Kernel size (3x3 or 5x5)
    activation_func = trial.suggest_categorical("activation_func", ["ReLU", "LeakyReLU"])
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])  # Use BatchNorm or not
    
    # Parameters for fully connected layers
    num_fc_layers = trial.suggest_int("num_fc_layers", 1, 3)  # Number of FC layers
    fc_layer_sizes = [trial.suggest_int(f"fc_size_{i+1}", 128, 1024, step=128) for i in range(num_fc_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)  # Dropout regularization rate
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Build convolutional stack dynamically
            conv_layers = []
            in_channels = 3
            for i, out_channels in enumerate(conv_filter_sizes):
                conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
                if use_batchnorm:
                    conv_layers.append(nn.BatchNorm2d(out_channels))
                if activation_func == "ReLU":
                    conv_layers.append(nn.ReLU())
                elif activation_func == "LeakyReLU":
                    conv_layers.append(nn.LeakyReLU(negative_slope=0.01))
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample
                in_channels = out_channels
            self.conv_stack = nn.Sequential(*conv_layers)

            # Calculate flattened size dynamically
            input_height, input_width = 224, 224  # Assuming input image size is 224x224
            for _ in range(num_conv_layers):
                input_height = input_height // 2
                input_width = input_width // 2
            flattened_size = conv_filter_sizes[-1] * input_height * input_width

            # Build fully connected stack dynamically
            fc_layers = []
            input_size = flattened_size
            for i, fc_size in enumerate(fc_layer_sizes):
                fc_layers.append(nn.Linear(input_size, fc_size))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rate))
                if use_batchnorm:
                    fc_layers.append(nn.BatchNorm1d(fc_size))
                input_size = fc_size
            fc_layers.append(nn.Linear(input_size, 2))  # Final output layer (7 classes)
            self.fc_stack = nn.Sequential(*fc_layers)

        def forward(self, x):
            x = self.conv_stack(x)  # Pass through convolutional layers
            x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers
            logits = self.fc_stack(x)  # Pass through fully connected layers
            return logits

    return Net()


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, verbose=False):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    num_batches = len(dataloader)

    for batch, (X,y) in tqdm(enumerate(dataloader),"training loop"):
        # Limit training data for faster epochs
        if batch * BATCHSIZE >= N_TRAIN_EXAMPLES:
            break

        # move X and y to the GPU!
        X,y = X.to(DEVICE), y.to(DEVICE)
        # compute prediction and loss!
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation!
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # compute training loss and accuracy
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # print message
        if batch % 100 and verbose:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print("predicted values: ",pred[:10].argmax(1))
            print("real values: ",y[:10])

    # print total train error
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    
    return 100*correct, train_loss



def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Limit validation data
            if batch * BATCHSIZE >= N_VALID_EXAMPLES:
                break

            # move X and y to the GPU!
            X,y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return 100*correct,test_loss


def objective(trial):
    try:
        # generate model based on trial parameters
        model = define_model(trial).to(DEVICE)

        # choose an optimizer as either (Adam, RMSprop, or SGD)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)   # Learning rate on a log scale
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Define data augmentation for training
        # flip_p = trial.suggest_float("flip", 0, 0.5) # define flip parameter!
        # random_rot = trial.suggest_int("random_rot", 10, 20)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), #flip_p),
            transforms.RandomVerticalFlip(p=0.5), #flip_p),
            transforms.RandomRotation(15), #random_rot),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define transform for testing (no augmentation, just normalization)
        test_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),  # Resize to a slightly larger fixed size
            transforms.CenterCrop(size=(224, 224)),  # Crop to the target size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # read in train and validation data
        data_path = os.path.join(os.getcwd(),"..","data")
        training_data = MNIST_skin_cancer(data_path,
                                        dataset_type="train",
                                        transform=train_transform
                                        )
        val_data = MNIST_skin_cancer(data_path,
                                    dataset_type="val",
                                    transform=test_transform
                                    )
        
        # define train and validation data_loader
        train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True,pin_memory=True, num_workers=8)
        val_dataloader = DataLoader(val_data, batch_size=BATCHSIZE, shuffle=True,pin_memory=True, num_workers=8)
        
        # Pass the class weights to CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss()

        # train and validate model for multiple epochs
        valid_accuracy_array = np.array([])
        valid_loss_array = np.array([])
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}\n--------------------------------------")
            train_accuracy,train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, BATCHSIZE, verbose=False)
            valid_accuracy,valid_loss = test_loop(val_dataloader, model, loss_fn)
            valid_accuracy_array = np.append(valid_accuracy_array, valid_accuracy)
            valid_loss_array = np.append(valid_loss_array, valid_loss)

        # prune trial if performance is poor
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return (np.mean(valid_accuracy_array[-10:])) # it gets stable in the end
    
    except torch.cuda.OutOfMemoryError: # when 8GB GPU RAM is not enough :(
        torch.cuda.empty_cache()        # clear the GPU memory!
        raise                           # re-raise exception so Optuna can catch it

if __name__ == "__main__":
    # Create an Optuna study to maximize validation accuracy
    # study = optuna.create_study(directions=["maximize","minimize"]) # multi-objective optimization
    study = optuna.create_study(direction="maximize")

    # Run the optimization with a timeout of 3000 seconds
    study.optimize(objective, n_trials=50, timeout=3600*2, catch=(torch.cuda.OutOfMemoryError,))

    # Analyze study results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Print statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)  # Best validation accuracy
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # plot param importances
    fig = vis.plot_param_importances(study)
    fig.write_image(os.path.join(DIR,"param_importances.png"))