import concurrent.futures
import os
import numpy as np 
from PIL import Image
from skimage.util import random_noise
from skimage.io import imread, imsave
import skimage
import random
path = "normal images"
labels = os.listdir(path)

#salt and paper noise
def noise(image_name):
    original_image = imread(path + "/" + image_name)
    noise_image = random_noise(original_image, mode = "gaussian")
    imsave("noised images/" + image_name, noise_image)


files =  os.listdir(path)
with concurrent.futures.ProcessPoolExecutor(max_workers = 15) as executor: # use 15 cores
	executor.map(noise, files)
#for single image
"""
def sp_noise(image_name):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob = 0.05
    image = Image.open(image_name)
    image = np.array(image)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    original_image = Image.fromarray(image)
    noise_image = Image.fromarray(output)
    original_image.save("orginal.png")
    noise_image.save("noise.png")
sp_noise("test.png")"""