import tensorflow as tf 
from skimage.io import imread, imshow
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
model = tf.keras.models.load_model("model.h5")
def load_img(path):

    img = imread(path, as_gray = True)
    #img = np.asarray(img, dtype = 'float32')
    img = np.expand_dims(img, axis = 0)
    img = np.expand_dims(img, axis = -1)
    img = img/255
    return img
img = load_img("noise.png")
result = model.predict(img)
result = np.squeeze(result, axis = 0)
result = np.squeeze(result, axis = -1)
result = result*255
img = Image.fromarray(result)
img.show()
img = img.convert("RGB")
img.save("result.png")