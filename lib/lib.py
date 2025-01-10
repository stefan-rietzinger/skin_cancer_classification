import sys,os
import numpy as np
import matplotlib.pyplot as plt

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
    ax[1].legend()

    for a in ax:
        a.grid()

    path = os.path.join(os.getcwd(),"results",folder_name,"plot.png")
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
    
