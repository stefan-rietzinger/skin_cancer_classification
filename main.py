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
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torchsummary import summary

import torchvision.models as models 

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

# class to import custom dataset
class MNIST_skin_cancer(Dataset):
    def __init__(self, data_path:str, train:bool, test_size:float=0.2, random_state:int=1, transform=None, target_transform=None):
        self.data_path = data_path
        self.img_dirs = [os.path.join(data_path,image_dirs) for image_dirs in ["ham10000_images_part_1","ham10000_images_part_2"]]
        self.transform = transform 
        self.target_transform = target_transform
        # split metadata in training and testing data! If training true use training data as metadata else use testing data
        metadata = pd.read_csv(os.path.join(data_path,"HAM10000_metadata.csv"))
        train_data, test_data = train_test_split(metadata, test_size=test_size, random_state=random_state, stratify=metadata["dx"])
        self.metadata = train_data if train else test_data
        # create dictionary to encode targets into integer values!
        self.class_to_idx = {label: idx for idx, label in enumerate(self.metadata['dx'].unique())}

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

# define Convolutionary Neural Network CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample
        )
        self.fc_stack = nn.Sequential(
            nn.Dropout(0.3),                   # regularization parameter!
            nn.Linear(64 * 56 * 75, 512), #1024),  #75!    # Adjusted for pooled output size
            nn.ReLU(),
            nn.BatchNorm1d(512), #1024),
            # nn.Dropout(0.2),
            nn.Linear(512,512), #1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            # nn.Dropout(0.2),
            nn.Linear(512, 7),  # Output layer
            nn.BatchNorm1d(7),
        )

    def forward(self, x):    
        x = self.pool(x)  # Pool on the original input
        x = self.conv_stack(x)  # Pass through convolutional stack
        x = torch.flatten(x, start_dim=1)  # Flatten before fully connected layers
        logits = self.fc_stack(x)  # Pass through fully connected layers
        return logits



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
    scheduler.step(loss)

def test_loop(dataloader, model, loss_fn, test_accuracy_list, test_loss_list):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            # move X and y to the GPU!
            X,y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

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
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transform for testing (no augmentation, just normalization)
    test_transform = transforms.Compose([
        # transforms.Resize(size=(256, 256)),  # Resize to a slightly larger fixed size
        # transforms.CenterCrop(size=(224, 224)),  # Crop to the target size
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read in train and test data
    data_path = os.path.join(os.getcwd(),"..","data")
    training_data = MNIST_skin_cancer(data_path,
                                      train=True,
                                      transform=train_transform
                                    )
    test_data = MNIST_skin_cancer(data_path,
                                  train=False, 
                                  transform=test_transform
                                )

    # define Hyperparameters!
    learning_rate = 1e-3
    batch_size = 64  
    epochs = 40
    momentum = 0.5
    weight_decay = 0.35

    # initialize DataLoaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=4)
 
    # create model and move to GPU
    # model_CNN = CNN().to(device)
    model_NN = Net().to(device)

    # initialize the loss function!
    # we have an inbalanced dataset, so I use weights to tackle the issue!
    y_train = torch.tensor([training_data.class_to_idx[dat] for dat in training_data.metadata["dx"]])
    # Compute class weights
    class_counts = torch.bincount(y_train)  # y_train contains your labels
    weights = 1. / class_counts.float()
    weights = weights.to(device)

    # Pass the class weights to CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # initialize the optimizer
    optimizer_NN = torch.optim.SGD(model_NN.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # initialize scheduler 
    # learning rate gets evaluated after every epoch!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NN, 'min', patience=5, verbose=True)

    # training model!
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
    start_time = time.time_ns()
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------------------")
        train_loop(train_dataloader, model_NN, loss_fn, optimizer_NN, scheduler, batch_size, train_accuracy, train_loss, verbose=False)
        test_loop(test_dataloader, model_NN, loss_fn, test_accuracy, test_loss)
    end_time = time.time_ns()

    print(f"DONE!, Execution time: {(end_time - start_time)/1e9} [s]")

    # print summary
    # summary(model_NN, input_size=(3, 224, 224))

    # plot accuracy and loss
    folder_name = "l2_regularization_7"
    lib.plot_loss_and_accuracy(epochs,train_accuracy,test_accuracy,train_loss,test_loss,folder_name)   

    # save accuracy and loss in .txt file 
    lib.save_training_results(train_accuracy,train_loss,test_accuracy,test_loss,folder_name)



# TODO:
# momentum einführen!
# learning rate erhoehen! (dadurch flexible learning rate abdrehen)
# Hyperparameters tunen!
# -> wenns nicht hoeher wird; binary outcome!