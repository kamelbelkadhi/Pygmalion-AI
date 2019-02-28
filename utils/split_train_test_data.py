import os
from tqdm import tqdm 
import pandas as pd 
from shutil import copyfile
import concurrent.futures

data_informations = pd.read_csv("children_data.csv")
#extract test data and put them in test fodler

def copy_data(file_name):
    file_path = "/home/kamel/Desktop/ChestXCaps/data (copy)/train/"
    file_serie = data_informations.loc[data_informations['Image Index'] == file_path] 
    print("ok")
    file_labels = file_serie["Finding Labels"]
    labels = str(file_labels.values)[2: -2]
    split_labels = labels.split(",")
    print(labels)
    for label in split_labels:
        copyfile(file_path + label + "/" + file_name, "../data/train/" + label + "/" + file_name)
def copy_data(file_name):
    file_path = "media/kamel/ADATA HD650/Data/Chest/" + file_name
    file_serie = data_informations.loc[data_informations['Image Index'] == file_name] 
    print("ok")
    file_labels = file_serie["Finding Labels"]
    labels = str(file_labels.values)[2: -2]
    split_labels = labels.split("|")
    for label in split_labels:
        copyfile(file_path[:-1], "../data/train/" + label + "/" + f[:-1] )    

test_files = open("test_list.txt")
train_files = open("train_val_list.txt")
"""
with concurrent.futures.ProcessPoolExecutor(max_workers = 15) as executor: # use 15 cores
    print("Copying test children's data ")
    executor.map(copy_data, test_files)"""

with concurrent.futures.ProcessPoolExecutor(max_workers = 15) as executor: # use 15 cores
    print("Copying train children's data ")
    executor.map(copy_data, train_files)
