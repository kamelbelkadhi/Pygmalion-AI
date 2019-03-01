import os
from PIL import Image
from tqdm import tqdm
import concurrent.futures

data_train_path = "normal images" 
def reshape(file_name):
    img=Image.open(data_train_path + "/" + file_name) 
    img=img.resize((128, 128)) 
    img.save("noised images/" + file_name)

files = os.listdir(data_train_path)
with concurrent.futures.ProcessPoolExecutor(max_workers = 15) as executor: # use 15 cores
	executor.map(reshape, files)