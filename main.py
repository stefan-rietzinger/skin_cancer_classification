import sys,os
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
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import torchvision.models as models 

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

class EarlyStopping:
    def __init__(self, patience:int=5, delta:float=0.0, verbose:bool=False):
        self.patience = patience 
        self.delta = delta 
        self.verbose = verbose 
        self.counter = 0
        self.best_score = None 
        self.early_stop = False 

    def __call__(self, val_loss):
        score = val_loss

        # if there is no best_score yet
        if self.best_score is None:
            self.best_score = score 

        # if score is worse than best score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStoppingCounter: {self.counter}")
            if self.counter > self.patience:
                # apply early stop
                if self.verbose:
                    print("Early stopping triggered!")
                self.early_stop = True 

        # if score is better than best_score
        else:
            self.best_score = score 
            self.counter = 0            


# # define Convolutionary Neural Network CNN
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 7)

#     def forward(self,x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv_stack = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample
#         )
#         self.fc_stack = nn.Sequential(
#             nn.Dropout(0.4),                   # regularization parameter!
#             nn.Linear(64 * 56 * 75, 512), #1024),  #75!    # Adjusted for pooled output size
#             nn.ReLU(),
#             nn.BatchNorm1d(512), #1024),
#             # nn.Dropout(0.2),
#             nn.Linear(512,512), #1024, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),

#             nn.Linear(512,256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),

#             # nn.Dropout(0.2),
#             nn.Linear(256, 7),  # Output layer
#             nn.BatchNorm1d(7),
#         )

#     def forward(self, x):    
#         x = self.pool(x)  # Pool on the original input
#         x = self.conv_stack(x)  # Pass through convolutional stack
#         x = torch.flatten(x, start_dim=1)  # Flatten before fully connected layers
#         logits = self.fc_stack(x)  # Pass through fully connected layers
#         return logits


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample

            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample
        )
        self.fc_stack = nn.Sequential(
            # nn.Linear(48 * 56 * 75, 1024),
            nn.Linear(16 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2779895419224205),

            # nn.Linear(640, 896), 
            # nn.Dropout(0.42068),

            # nn.Linear(896, 384),
            # nn.ReLU(),
            # nn.Dropout(0.42068),

            nn.Linear(1024, 2),  # Output layer
        )

    def forward(self, x):    
        x = self.pool(x)  # Pool on the original input
        x = self.conv_stack(x)  # Pass through convolutional stack
        # print(f"Shape after conv_stack: {x.shape}")
        x = torch.flatten(x, start_dim=1)  # Flatten before fully connected layers
        logits = self.fc_stack(x)  # Pass through fully connected layers
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, batch_size, train_accuracy_list, train_loss_list, verbose=False):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    num_batches = len(dataloader)

    for batch, (X,y) in tqdm(enumerate(dataloader),"training loop"):
        # move X and y to the GPU!
        X,y = X.to(device), y.to(device)
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
    
    # append to lists
    train_accuracy_list.append(100*correct)
    train_loss_list.append(train_loss)

    # update scheduler after each epoch!
    # scheduler.step(loss)

def test_loop(dataloader, model, loss_fn, test_accuracy_list, test_loss_list, verbose=True):
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
            # move X and y to the GPU!
            X,y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # print message
            if batch % 100 and verbose:
                print("predicted values: ",pred[:10].argmax(1))
                print("real values: ",y[:10])

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # append to lists
    test_accuracy_list.append(100*correct)
    test_loss_list.append(test_loss)

if __name__=="__main__":
    # define device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mos" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # data augmentation!
    # Define data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transform for testing (no augmentation, just normalization)
    test_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),  # Resize to a slightly larger fixed size
        transforms.CenterCrop(size=(224, 224)),  # Crop to the target size
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read in train and test data
    data_path = os.path.join(os.getcwd(),"..","data")
    training_data = MNIST_skin_cancer(data_path,
                                      dataset_type="train",
                                      transform=train_transform
                                    )
    test_data = MNIST_skin_cancer(data_path,
                                  dataset_type="test",
                                  transform=test_transform
                                )

    # define Hyperparameters!
    learning_rate = 0.00014508464989590903 # 0.00037628287996986865 # 1e-3
    batch_size = 128 
    epochs = 100
    # momentum = 0.5
    weight_decay = 2.4467806756368742e-05 # 0.00037628287996986865 #1e-4 # 0.35

    # initialize DataLoaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=4)
 
    # create model and move to GPU
    # model_CNN = CNN().to(device)
    model_NN = Net().to(device)

    # initialize the loss function!
    # we have an inbalanced dataset, so I use weights to tackle the issue!
    # y_train = torch.tensor([training_data.class_to_idx[dat] for dat in training_data.metadata["dx"]])
    # # Compute class weights
    # class_counts = torch.bincount(y_train)  # y_train contains your labels
    # weights = 1. / class_counts.float()
    # weights = weights.to(device)
    # print(weights)

    # Pass the class weights to CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss() #weight=weights)

    # initialize the optimizer
    # optimizer_NN = torch.optim.SGD(model_NN.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # optimizer_NN = torch.optim.SGD(model_NN.parameters(), lr=learning_rate) 
    optimizer_NN = torch.optim.Adam(model_NN.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # initialize scheduler 
    # learning rate gets evaluated after every epoch!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NN, 'min', patience=5)
    
    # Initialize early stopping
    early_stopper = EarlyStopping(patience=7, verbose=True)

    # training model!
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
    start_time = time.time_ns()
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------------------")
        train_loop(train_dataloader, model_NN, loss_fn, optimizer_NN, scheduler, batch_size, train_accuracy, train_loss, verbose=False)
        test_loop(test_dataloader, model_NN, loss_fn, test_accuracy, test_loss, verbose=False)
        
        # check for early stopping
        early_stopper(test_loss[-1])
        if early_stopper.early_stop: 
            print(f"Early stopping was triggered at epoch: {t}")
            break

        # step of learning rate using scheduler
        scheduler.step(test_loss[-1])

    end_time = time.time_ns()

    print(f"DONE!, Execution time: {(end_time - start_time)/1e9} [s]")

    # plot accuracy and loss
    folder_name = "binary_outcome_test"
    lib.plot_loss_and_accuracy(t+1,train_accuracy,test_accuracy,train_loss,test_loss,folder_name)   

    # print and save summary
    lib.save_model_summary(model_NN, folder_name)

    # save accuracy and loss in .txt file 
    lib.save_training_results(train_accuracy,train_loss,test_accuracy,test_loss,folder_name)

    # save and plot model diagnostics!
    lib.plot_confusion_matrix(model_NN, test_dataloader, device, folder_name)
    lib.save_classification_report(model_NN, test_dataloader, device, folder_name)

# TODO:
# momentum einführen!
# learning rate erhoehen! (dadurch flexible learning rate abdrehen)
# Hyperparameters tunen!
# -> wenns nicht hoeher wird; binary outcome!


# Funktionenfriedhof:
# ----------------------------------


# class Net(nn.Module): # custom CNN model
    
#     def __init__(self, model='resnet50'):
#         super(Net, self).__init__()
        
#         # self.num_classes = num_classes
#         # print(f'There are {self.num_classes} classes.')
#         self.num_classes = 7
        
#         self.chosen_model = model
        
#         # Choosing backbone CNN
            
#         self.model = models.resnet50(pretrained=True)
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.1), # regularisation
            
#             # Common practise to set nn.Linear with no bias if using batchnorm after it.
#             # As batchnorm normalises activitations from nn.Linear, it also removes the bias from nn.Linear,
#             # and it adds its own bias term. Thus, nn.Linear bias term is redundant.
            
#             nn.Linear(self.model.fc.in_features, 256, bias=False), 
#             nn.ReLU(),
#             nn.BatchNorm1d(256),

#             nn.Linear(256, 128, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),

#             nn.Linear(128, self.num_classes, bias=False),
#             nn.BatchNorm1d(self.num_classes), 
#         )
#         self.model.fc = self.classifier
        
 
#         print(f'{self.chosen_model} created')
        
#         model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         print(f'Model has {params} trainable params.')

#     def forward(self, x):
        
#         return self.model(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv_stack = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc_stack = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
#             nn.Flatten(),
#             nn.Linear(64, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),  # Dropout for regularization
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 7),  # Output layer for 7 classes
#         )

#     def forward(self, x):
#         x = self.conv_stack(x)
#         x = self.fc_stack(x)
#         return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Linear(3*450*600, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024,1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024,512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512,512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512,7),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


# # definition of a convolutional NN
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv_stack = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample
#         )
#         self.fc_stack = nn.Sequential(
#             nn.Linear(128 * 28 * 37, 512),  # Adjusted for pooled output size
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(512,512),
#             nn.ReLU(),
#             # nn.Dropout(0.05),
#             # nn.Linear(1024, 512),
#             # nn.ReLU(),
#             # nn.Dropout(0.05),
#             # nn.Linear(512, 512),
#             # nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(512,7) # output layer
#         )

#     def forward(self, x):
#         x = self.pool(x)  # Pool on the original input
#         x = self.conv_stack(x)  # Pass through convolutional stack
#         x = torch.flatten(x, start_dim=1)  # Flatten before fully connected layers
#         logits = self.fc_stack(x)  # Pass through fully connected layers
#         return logits

# class MNIST_skin_cancer(Dataset):
#     def __init__(self, data_path: str, train: bool, test_size: float = 0.2, random_state: int = 1, transform=None, target_transform=None):
#         self.data_path = data_path
#         self.img_dirs = [os.path.join(data_path, image_dirs) for image_dirs in ["ham10000_images_part_1", "ham10000_images_part_2"]]
#         self.transform = transform 
#         self.target_transform = target_transform
#         # split metadata in training and testing data! If training true use training data as metadata else use testing data
#         metadata = pd.read_csv(os.path.join(data_path, "HAM10000_metadata.csv"))
#         train_data, test_data = train_test_split(metadata, test_size=test_size, random_state=random_state, stratify=metadata["dx"])
#         self.metadata = train_data if train else test_data
        
#         # create dictionary to encode targets into integer values!
#         self.class_to_idx = {label: idx for idx, label in enumerate(self.metadata['dx'].unique())}
#         self.num_classes = len(self.class_to_idx)  # Number of classes (for one-hot encoding)

#     def __len__(self) -> int:
#         return len(self.metadata)
    
#     def __getitem__(self, idx: int) -> tuple:
#         # read out label and image name from metadata!
#         label = self.metadata.iloc[idx].dx
#         image_name = f"{self.metadata.iloc[idx].image_id}.jpg"

#         # find image file in img_dirs and read into image!
#         for directory in self.img_dirs:
#             img_path = os.path.join(self.data_path, directory, image_name)
#             if os.path.exists(img_path):
#                 image = read_image(img_path).float()
#                 break
#         else:
#             raise Exception("FILE DOES NOT EXIST!")
        
#         # transform label into numerical value
#         label = self.class_to_idx[label]
        
#         # One-hot encode the label
#         label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)  # One-hot encoding

#         # apply transform or target transform if wanted
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label