import torch 
import kagglehub
import lib.lib as lib

if __name__=="__main__":

    # download latest version of dataset!
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print("Path to dataset files:", path)
