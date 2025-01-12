import sys,os
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from io import StringIO
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch

def plot_loss_and_accuracy(epochs,train_accuracy,test_accuracy,train_loss,test_loss,folder_name):
        # plot train and test accuracy 
    fig,ax = plt.subplots(1,2,figsize=(14,6))
    x = np.arange(1,epochs+1)
    ax[0].plot(x,train_accuracy,"r+-",linewidth=1,markersize=15, label="train accuracy")
    ax[0].plot(x,test_accuracy,"b+-",linewidth=1,markersize=15, label="test accuracy")
    ax[0].set_title("accuracy over epochs")
    ax[0].set_ylabel("accuracy [%]")
    ax[0].set_ylim(0,100)
    ax[0].set_xlabel("epochs")
    ax[0].legend()

    ax[1].plot(x,train_loss,"r+-",linewidth=1,markersize=15, label="train loss")
    ax[1].plot(x,test_loss,"b+-",linewidth=1,markersize=15, label="test loss")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("cross entropy loss")
    ax[1].set_title("cross entropy loss over epochs")
    ax[1].set_ylim(0,1)
    ax[1].legend()

    for a in ax:
        a.grid()

    path = os.path.join(os.getcwd(),"results",folder_name,"accuracy_over_epochs.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.show()

def save_training_results(train_accuracy, train_loss, test_accuracy, test_loss, folder_name):

    path = os.path.join(os.getcwd(),"results",folder_name,"info.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        f.write("Training Results\n")
        f.write("=" * 20 + "\n\n")
        f.write("Train Accuracy:\n")
        f.write(", ".join(map(str, train_accuracy)) + "\n\n")
        f.write("Train Loss:\n")
        f.write(", ".join(map(str, train_loss)) + "\n\n")
        f.write("Test Accuracy:\n")
        f.write(", ".join(map(str, test_accuracy)) + "\n\n")
        f.write("Test Loss:\n")
        f.write(", ".join(map(str, test_loss)) + "\n")

def save_model_summary(model, folder_name):
    path = os.path.join(os.getcwd(),"results",folder_name,"model_summary.txt")  

    # Redirect std out to buffer and save it
    original_stdout = sys.stdout
    with StringIO() as buffer:
        sys.stdout = buffer
        summary(model, input_size=(3, 224, 224))
        sys.stdout = original_stdout
        summary_str = buffer.getvalue()

    # now save string
    with open(path, "w") as f:
        f.write(summary_str) 

def compute_y_pred_and_y(model, test_dataloader, device):
    
    # ensure model is in evaluation mode
    model.eval()

    # initialize arrays to store true and predicted values
    y_true, y_pred = np.array([]), np.array([])

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            # Move data to the same device as the model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Get model predictions (and transform them into np.arrays!)
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, axis=1).cpu().numpy()
            y = y_batch.cpu().numpy()

            # append values to y_true and y_pred
            y_true = np.append(y_true, y)
            y_pred = np.append(y_pred, predictions)

    return y_true,y_pred

def plot_confusion_matrix(model, test_dataloader, device, folder_name):
    y_true, y_pred = compute_y_pred_and_y(model, test_dataloader, device)
    cf_matrix = confusion_matrix(y_true,y_pred)

    fig,ax = plt.subplots(figsize=(12,8))
    
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    path = os.path.join(os.getcwd(),"results",folder_name,"confusion_matrix.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.show()

def save_classification_report(model, test_dataloader, device, folder_name):

    y_true,y_pred = compute_y_pred_and_y(model, test_dataloader, device)
    model_report = classification_report(y_true, y_pred)

    path = os.path.join(os.getcwd(),"results",folder_name,"model_report.txt")  
    # now save string
    with open(path, "w") as f:
        f.write(model_report) 

