# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:29:42 2021

@author: mario
"""

import keras
keras.__version__

from keras.models import load_model
model = load_model('fer_model_1m.h5')
model.summary()  # As a reminder.

img_path = "C:/Users/mario/Documents/coding_TA/dataset_1m/train/terkejut/frame729.jpg"

# -----------------------------------------------------------------------

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np
import cv2 

IMAGE_SIZE = (120,120)
# Image filters for sharpening
filters = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Image Processing with OpenCV
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMAGE_SIZE)
img = cv2.filter2D(img,-1, filters)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 120, 120, 3)
print(img_tensor.shape)

# Showing Imported Image
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

# -----------------------------------------------------------------------
from keras import models

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)


first_layer_activation = activations[0]
print(first_layer_activation.shape)


import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()


plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()


import keras

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()

