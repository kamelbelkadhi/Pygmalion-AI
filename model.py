import numpy as np
import tensorflow as tf
from skimage.io import imread
import random
import os
from PIL import Image
image_dim = 128


def load_img(path):

    img = imread(path, as_gray = True)
    #img = np.asarray(img, dtype = 'float32')
    img = np.expand_dims(img, axis = 0)
    img = np.expand_dims(img, axis = -1)
    img = img/255
    return img

def get_img(path):
    img = imread(path, as_gray = True)
    img = np.expand_dims(img, axis = -1)
    return(img)

def normalize_image(img):
    img_normalized  = img/255
    return img_normalized

def image_generator(input_path, output_path, batch_size = 64):
    images = os.listdir(input_path)
    while True:
          # Select files (paths/indices) for the batch
          batch_paths = np.random.choice(a = images, 
                                         size = batch_size)
          batch_input = []
          batch_output = [] 
          
          # Read in each input, perform preprocessing and get labels
          for image in batch_paths:
              input = get_img(input_path + "/" + image)
              output = get_img(output_path + "/" + image)
            
              input = normalize_image(img = input)
              output = normalize_image(img = output)
              batch_input += [ input ]
              batch_output += [ output ]
          # Return a tuple of (input,output) to feed the network
          batch_x = np.array( batch_input )
          batch_y = np.array( batch_output )
        
          yield( batch_x, batch_y )

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 3
    if epoch % decay_step == 0 and epoch != 0:
        return lr * decay_rate
    return lr

def get_model():
    img_encoder_input = tf.keras.layers.Input(shape=(image_dim, image_dim, 1))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_encoder_input) 
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) 
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) 
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2) 
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2) 
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    #decoder
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) 
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    up1 = tf.keras.layers.UpSampling2D((2,2))(conv4) 
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    up2 = tf.keras.layers.UpSampling2D((2,2))(conv5) 
    conv6 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2) 
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    up6 = tf.keras.layers.UpSampling2D((2,2))(conv6)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up6)
    autoencoder = tf.keras.models.Model(img_encoder_input, decoded)
    return autoencoder

save_results = tf.keras.callbacks.CSVLogger(("results.csv"), separator=',', append=True)
save_model = tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=3)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = "loss", patience = 3, verbose=1, factor = 0.1)
model = get_model()
model.summary()
model.compile(loss='mean_squared_error', optimizer = 'RMSprop')
model = tf.keras.models.load_model("model.h5")
generator = image_generator(input_path = "noised images", output_path = "normal images", batch_size = 32)
k = 0
model.fit_generator(generator, epochs= 1000, steps_per_epoch = 500, use_multiprocessing = True, workers = -1, callbacks = [save_results, save_model, lr])
"""
while True:
  model.fit_generator(generator, epochs= 1, steps_per_epoch = 50, use_multiprocessing = True, workers = -1, callbacks = [save_results, save_model, lr])
  img = load_img("noise.png")
  result = model.predict(img)
  result = np.squeeze(result, axis = 0)
  result = np.squeeze(result, axis = -1)
  img = np.squeeze(img, axis = 0)
  img = np.squeeze(img, axis = -1)
  for i in range(0, img.shape[0]):
      for j in range(0, img.shape[1]):
          img[i][j] = result[i][j]*255
  img = Image.fromarray(img)
  img = img.convert("RGB")
  img.save("images/result_" + str(k) + ".png")
  print(str(k))
  k = k + 1"""